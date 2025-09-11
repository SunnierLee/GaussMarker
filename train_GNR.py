import os
import argparse
import logging
from tqdm import tqdm
import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from watermark import *
from unet.unet_model import UNet

def flip_tensor(tensor, flip_prob):
    random_tensor = torch.rand(tensor.size())
    flipped_tensor = tensor.clone()
    flipped_tensor[random_tensor < flip_prob] = 1 - flipped_tensor[random_tensor < flip_prob]
    return flipped_tensor

def Affine_random(latent, r, t, s_min, s_max, sh):
    config = dict(degrees=(-r, r), translate=(t, t), scale_ranges=(s_min, s_max), shears=(-sh, sh), img_size=latent.shape[-2:])
    r, (tx, ty), s, (shx, shy) = transforms.RandomAffine.get_params(**config)
    
    b, c, w, h = latent.shape
    new_latent = transforms.functional.affine(latent.view(b*c, 1, w, h), angle=r, translate=(tx, ty), scale=s, shear=(shx, shy), fill=999999)
    new_latent = new_latent.view(b, c, w, h)

    mask = (new_latent[:, :1, ...] < 999998).float()
    new_latent = new_latent * mask + torch.randint_like(new_latent, low=0, high=2) * (1-mask)
    
    return new_latent, (r, tx, ty, s)

class LatentDataset_m(IterableDataset):
    def __init__(self, watermark, args):
        super(LatentDataset_m, self).__init__()
        self.watermark = watermark
        self.args = args
        if self.args.num_watermarks > 1:
            t_m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)
            o_m = torch.randint(low=0, high=2, size=(self.args.num_watermarks-1, 4, 64, 64))
            self.m = torch.cat([t_m, o_m])
        else:
            self.m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)
        self.args.neg_p = 1 / (1 + self.args.num_watermarks)
    
    def __iter__(self):
        while True:
            random_index = torch.randint(0, self.args.num_watermarks, (1,)).item()
            latents_m = self.m[random_index:random_index+1]
            false_latents_m = torch.randint_like(latents_m, low=0, high=2)
            # latents_m = latents_m[:, :1, ...]
            # false_latents_m = false_latents_m[:, :1, ...]
            if np.random.rand() > self.args.neg_p:
                aug_latents_m, params = Affine_random(latents_m.float(), self.args.r, self.args.t, self.args.s_min, self.args.s_max, self.args.sh)
                aug_latents_m = flip_tensor(aug_latents_m, self.args.fp)
                yield aug_latents_m.squeeze(0).float(), latents_m.squeeze(0).float()
            else:
                aug_false_latents_m, params = Affine_random(false_latents_m.float(), self.args.r, self.args.t, self.args.s_min, self.args.s_max, self.args.sh)
                aug_false_latents_m = flip_tensor(aug_false_latents_m, self.args.fp)
                yield aug_false_latents_m.squeeze(0).float(), aug_false_latents_m.squeeze(0).float()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    gfile_stream = open(os.path.join(args.output_path, 'log.txt'), 'a')
    set_logger(gfile_stream)
    logging.info(args)

    num_steps = args.train_steps
    bs = args.batch_size

    model = UNet(4, 4, nf=args.model_nf).cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters in model: %d' % n_params)
    logging.info('Number of trainable parameters in model: %d' % n_params)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if os.path.exists(args.w_info_path):
        w_info = torch.load(args.w_info_path)
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"], nonce=w_info["nonce"])
    else:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number)
        _ = watermark.create_watermark_and_return_w_m()
        torch.save({"w": watermark.watermark, "m": watermark.m, "key": watermark.key, "nonce": watermark.nonce}, args.w_info_path)

    if args.sample_type == "m":
        dataset = LatentDataset_m(watermark, args)
    else:
        raise NotImplementedError
    
    data_loader = DataLoader(dataset, batch_size=bs, num_workers=args.num_workers)

    for i, batch in tqdm(enumerate(data_loader)):
        x, y = batch
        # print(x[0, 0])
        x = x.cuda()
        y = y.cuda().float()

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 2000 == 0:
        #     torch.save(model.state_dict(), os.path.join(args.output_path, "model_{}.pth".format(i)))
        if i % 2000 == 0:
            # torch.save(model.state_dict(), os.path.join(args.output_path, "model_{}.pth".format(i)))
            pred = F.sigmoid(pred)
            save_imgs = torch.cat([x[:, :1, ...].unsqueeze(0), pred[:, :1, ...].unsqueeze(0), y[:, :1, ...].unsqueeze(0)]).permute(1, 0, 2, 3, 4).contiguous()
            save_imgs = save_imgs.view(-1, save_imgs.shape[2], save_imgs.shape[3], save_imgs.shape[4])[:64]
            save_image(save_imgs, os.path.join(args.output_path, "sample_{}.png".format(i)), nrow=6)
        if i % 200 == 0:
            print(loss.item())
            torch.save(model.state_dict(), os.path.join(args.output_path, "checkpoint.pth".format(i)))
            logging.info("Iter {} Loss {}".format(i, loss.item()))

        if i > num_steps:
            break
    
    torch.save(model.state_dict(), os.path.join(args.output_path, "model_final.pth"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--w_copy', default=8, type=int)
    parser.add_argument('--h_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./GNR')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--w_info_path', default='./w1.pth')

    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample_type', default="m")
    parser.add_argument('--r', type=float, default=8)
    parser.add_argument('--t', type=float, default=0)
    parser.add_argument('--s_min', type=float, default=0.5)
    parser.add_argument('--s_max', type=float, default=2.0)
    parser.add_argument('--sh', type=float, default=0)
    parser.add_argument('--fp', type=float, default=0.00)
    parser.add_argument('--neg_p', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_watermarks', type=int, default=1)

    parser.add_argument('--model_nf', type=int, default=64)
    parser.add_argument('--exp_description', '-ed', default="")

    args = parser.parse_args()

    # multiprocessing.set_start_method("spawn")
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # args.output_path = args.output_path + 'r{}_t{}_s_{}_{}_sh{}_fp{}_np{}_{}_{}'.format(args.r, args.t, args.s_min, args.s_max, args.sh, args.fp, args.neg_p, args.exp_description, nowTime)
    args.output_path = args.output_path + '_' + args.exp_description

    main(args)
