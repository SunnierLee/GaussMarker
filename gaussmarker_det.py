import os
import argparse
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from sklearn import metrics
import numpy as np
import logging
from scipy.special import betainc
import joblib

# from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from watermark import Gaussian_Shading_chacha, Gaussian_Shading
from utils import Affine, measure_similarity, image_distortion, transform_img, set_random_seed

from tr_utils import get_watermarking_mask, eval_watermark
from unet.unet_model import UNet

def set_logger(workdir, args):
    os.makedirs(workdir, exist_ok=True)
    gfile_stream = open(os.path.join(workdir, 'log.txt'), 'a')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logging.info(args)



class Evaluator(object):
    def __init__(self, ch_factor, w_factor, h_factor, fpr, user_number):
        self.ch = ch_factor
        self.w = w_factor
        self.h = h_factor
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.w * self.h)
        self.tau_onebit = None
        self.tau_bits = None
        self.fpr = fpr

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
    
    def eval(self, ws, preds):
        tp_onebit_count = 0
        tp_bits_count = 0
        accs = []
        for i in range(len(ws)):
            w = ws[i]
            pred = preds[i]
            correct = (pred == w).float().mean().item()
            if correct >= self.tau_onebit:
                tp_onebit_count = tp_onebit_count+1
            if correct >= self.tau_bits:
                tp_bits_count = tp_bits_count + 1
            accs.append(correct)
        return accs, tp_onebit_count, tp_bits_count

    def eval2(self, ws, w_preds, no_ws, no_w_preds):
        t_labels = []
        preds = []
        for i in range(len(ws)):
            w = ws[i]
            pred = w_preds[i]
            correct = (pred == w).float().mean().item()
            t_labels.append(1)
            preds.append(correct)
        bit_acc = np.mean(preds)
        for i in range(len(no_ws)):
            w = no_ws[i]
            pred = no_w_preds[i]
            correct = (pred == w).float().mean().item()
            t_labels.append(0)
            preds.append(correct)
        print(preds[:len(ws)])
        print(preds[-len(no_ws):])
        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        low = tpr[np.where(fpr<self.fpr)[0][-1]]
        thre = thresholds[np.where(fpr<self.fpr)[0][-1]]
        return auc, acc, low, thre, bit_acc

    def eval_ring(self, no_w_metrics_affine, w_metrics_affine):
        preds_affine = no_w_metrics_affine +  w_metrics_affine
        t_labels_affine = [0] * len(no_w_metrics_affine) + [1] * len(w_metrics_affine)

        fpr, tpr, thresholds = metrics.roc_curve(t_labels_affine, preds_affine, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        low = tpr[np.where(fpr<args.fpr)[0][-1]]
        thre = thresholds[np.where(fpr<self.fpr)[0][-1]]
        return auc, acc, low, thre
    
    def eval_ensemble(self, ws, w_preds, no_ws, no_w_preds, no_w_metrics_affine, w_metrics_affine, clf=None):
        x = []
        y = []
        for i in range(len(ws)):
            w = ws[i]
            pred = w_preds[i]
            correct = (pred == w).float().mean().item()
            x.append([correct, w_metrics_affine[i]])
            y.append(1)
        for i in range(len(no_ws)):
            w = no_ws[i]
            pred = no_w_preds[i]
            correct = (pred == w).float().mean().item()
            x.append([correct, no_w_metrics_affine[i]])
            y.append(0)
        x = np.array(x)
        y = np.array(y)

        preds = clf.predict_proba(x)[:, 1].tolist()

        fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        low = tpr[np.where(fpr<self.fpr)[0][-1]]
        thre = thresholds[np.where(fpr<self.fpr)[0][-1]]
        return auc, acc, low, thre, clf


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    os.makedirs(args.output_path, exist_ok=True)

    # class for watermark
    w1_path = args.w1_path
    w2_path = args.w2_path
    if args.chacha:
        if os.path.exists(w1_path):
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"], nonce=w_info["nonce"])
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)
    else:
        #a simple implement,
        if os.path.exists(w1_path):
            w_info = torch.load(w1_path)
            watermark = Gaussian_Shading(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"])
        m = torch.from_numpy(watermark.m).reshape(1, 4, 64, 64)
    
    evaluator = Evaluator(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    #acc
    acc = []
    #CLIP Scores
    clip_scores = []

    pred_gs = []
    pred_tr = []
    pred_restore = []
    pred_gs_neg = []
    pred_tr_neg = []
    pred_restore_neg = []
    w_list = []

    if os.path.exists(w2_path):
        gt_patch = torch.load(w2_path).to(device)
    watermarking_mask = get_watermarking_mask(gt_patch.real, args, device)

    if args.GNR_path is not None:
        model = UNet(8 if args.classifier_type == 1 else 4, 4, nf=args.model_nf).cuda()
        model.eval()
        model.load_state_dict(torch.load(args.GNR_path))
    else:
        model = None
    clf_2 = joblib.load('sd21_cls2.pkl')

    #test
    for i in tqdm(range(args.num)):
        if os.path.exists(os.path.join(args.output_path, 'pred_res.pth')):
            break
        seed = i + args.gen_seed

        #generate with watermark
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()

        if args.advanced_attack is None:
            image_w = Image.open(os.path.join(args.input_path, "w_img", "{}.png".format(i)))
        else:
            if os.path.exists(os.path.join(args.input_path, args.advanced_attack, "{}.png".format(i))):
                image_w = Image.open(os.path.join(args.input_path, args.advanced_attack, "{}.png".format(i)))
            else:
                break
        if args.no_w_path is None:
            image_no_w = Image.open(os.path.join(args.input_path, "no_w_img", "{}.png".format(i)))
        else:
            image_no_w = Image.open(os.path.join(args.no_w_path, "{}.png".format(i)))
        with open(os.path.join(args.input_path, "prompt", "{}.txt".format(i)), 'r', encoding='utf-8') as f:
            current_prompt = f.read()
        
        w_list.append(watermark.watermark)

        # distortion
        while True:
            try:
                image_w_distortion = image_distortion(image_w, seed, args)
                image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
                image_no_w_distortion = image_distortion(image_no_w, seed, args)
                image_no_w_distortion = transform_img(image_no_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
                break
            except:
                pass

        with torch.no_grad():

            if args.affine is not None:
                affine_image_w_distortion, affine_image_latents_w, ori_image_latents_no_w, params, trans_m = Affine(image_w_distortion, pipe.get_image_latents, m.float(), args.affine_params)
                affine_image_no_w_distortion, affine_image_latents_no_w, ori_image_latents_no_w, _, _ = Affine(image_no_w_distortion, pipe.get_image_latents, None, args.affine_params)
            else:
                affine_image_latents_w = pipe.get_image_latents(image_w_distortion)
                affine_image_latents_no_w = pipe.get_image_latents(image_no_w_distortion)

            reversed_latents_w_affine = pipe.forward_diffusion(
                latents=affine_image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inversion_steps,
            )
            reversed_latents_no_w_affine = pipe.forward_diffusion(
                latents=affine_image_latents_no_w,
                text_embeddings=text_embeddings,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inversion_steps,
            )

            reversed_m_affine = (reversed_latents_w_affine > 0).float().cpu()
            reversed_no_m_affine = (reversed_latents_no_w_affine > 0).float().cpu()
            if args.affine is not None:
                trans_m = (trans_m > 0.5).int()
                print("synthe acc: ", (trans_m==reversed_m_affine.int()).float().mean())
            
            w = watermark.pred_w_from_latent(reversed_latents_w_affine)
            pred_gs.append(w)
            w = watermark.pred_w_from_latent(reversed_latents_no_w_affine)
            pred_gs_neg.append(w)

            restored_reversed_m = (F.sigmoid(model(torch.cat([m.float(), reversed_no_m_affine], dim=1).cuda())) > 0.5).int() if args.classifier_type == 1 else (F.sigmoid(model(reversed_no_m_affine.cuda())) > 0.5).int()
            w = watermark.pred_w_from_m(restored_reversed_m)
            pred_restore_neg.append(w)

            restored_reversed_m = (F.sigmoid(model(torch.cat([m.float(), reversed_m_affine], dim=1).cuda())) > 0.5).int().cpu() if args.classifier_type == 1 else (F.sigmoid(model(reversed_m_affine.cuda())) > 0.5).int()
            w = watermark.pred_w_from_m(restored_reversed_m)
            pred_restore.append(w)

            no_w_metric_affine, w_metric_affine = eval_watermark(reversed_latents_no_w_affine, reversed_latents_w_affine, watermarking_mask, gt_patch, args)
            pred_tr_neg.append(-no_w_metric_affine)
            pred_tr.append(-w_metric_affine)

            if args.reference_model is not None:
                socre = measure_similarity([image_w], current_prompt, ref_model,
                                                ref_clip_preprocess,
                                                ref_tokenizer, device)
                clip_socre = socre[0].item()
            else:
                clip_socre = 0
            clip_scores.append(clip_socre)



    #tpr metric
    # acc, tpr_detection, tpr_traceability = evaluator.eval(w_list, pred_affine)
    # save_metrics(args, tpr_detection, tpr_traceability, acc, None)

    # acc, tpr_detection, tpr_traceability = evaluator.eval(w_list, pred_affine_restore)
    # save_metrics(args, tpr_detection, tpr_traceability, acc, None)

    # acc, tpr_detection, tpr_traceability = evaluator.eval(w_neg_list, pred_affine)
    # save_metrics(args, tpr_detection, tpr_traceability, acc, None)

    # acc, tpr_detection, tpr_traceability = evaluator.eval(w_neg_list, pred_affine_neg_restore)
    # save_metrics(args, tpr_detection, tpr_traceability, acc, None)
    if os.path.exists(os.path.join(args.output_path, 'pred_res.pth')):
        pred_res = torch.load(os.path.join(args.output_path, 'pred_res.pth'))
        w_list = pred_res['w_list']
        pred_gs = pred_res['pred_gs_w']
        pred_gs_neg = pred_res['pred_gs_no_w']
        pred_restore = pred_res['pred_r_w']
        pred_restore_neg = pred_res['pred_r_no_w']
        pred_tr = pred_res['pred_tr_w']
        pred_tr_neg = pred_res['pred_tr_no_w']
    else:
        pred_res = {
            "w_list": w_list,
            "pred_gs_w": pred_gs,
            "pred_gs_no_w": pred_gs_neg,
            "pred_r_w": pred_restore,
            "pred_r_no_w": pred_restore_neg,
            "pred_tr_w": pred_tr,
            "pred_tr_no_w": pred_tr_neg,
            }
        torch.save(pred_res, os.path.join(args.output_path, "pred_res.pth"))

    auc, acc, low, thre, _ = evaluator.eval_ensemble(w_list, pred_restore, w_list, pred_restore_neg, pred_tr_neg, pred_tr, clf_2)
    logging.info(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}, Threshold: {thre}')
    # auc, acc, low, thre = evaluator.eval_ensemble(w_list, pred_gs, w_list, pred_gs_neg, pred_tr_neg, pred_tr)
    # logging.info('GS+TR: '+f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}, Threshold: {thre}')
    # logging.info(f'clip score mean: {np.mean(clip_scores)} std: {np.std(clip_scores)}')


if __name__ == '__main__':
    def parse_floats(string):
        try:
            return [[float(x) for x in string_i.split(',')] for string_i in string.split('-')]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float value: {string}")
    parser = argparse.ArgumentParser(description='GaussMarker')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=1., type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--w_copy', default=8, type=int)
    parser.add_argument('--h_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.01, type=float)
    parser.add_argument('--input_path', default='./gr_gen_1000_sd21')
    parser.add_argument('--output_path', default='./output_EUattack_num10/')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--no_w_path', default=None)
    parser.add_argument('--GNR_path', default=None)
    parser.add_argument('--classifier_type', default=0, type=int)
    parser.add_argument('--w1_path', default='w1.pth')
    parser.add_argument('--w2_path', default='w2.pth')

    # for image distortion
    parser.add_argument('--advanced_attack', default=None, type=str)
    parser.add_argument('--affine', default=None, type=float)
    parser.add_argument('--affine_params', default=(8, 0, 1., 0), type=parse_floats)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)

    parser.add_argument('--model_nf', type=int, default=128)

    # treering
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=3, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=4, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
    set_logger(args.output_path, args)
    all_affine_params = args.affine_params
    work_dir = args.output_path
    if args.affine is not None:
        for affine_params in all_affine_params:
            logging.info("Affine params: " + str(affine_params))
            args.affine_params = affine_params
            # args.output_path = os.path.join(work_dir, str(affine_params))
            main(args)
    else:
        main(args)

    # main(args)