import torch
from datasets import load_dataset
import json
import numpy as np
import os
from torchvision import transforms

from PIL import Image, ImageFilter
import random


def get_dataset(args):
    if 'laion' in args.dataset_path:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset_path:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset_path)['train']
        prompt_key = 'Prompt'
    return dataset, prompt_key


def Affine(watermarked_image, Encoder, init_latent, params):
    r, t, s, sh = params
    t = (watermarked_image.shape[-2]*t, watermarked_image.shape[-1]*t)
    # r = (np.random.rand() * 2 - 1) * r
    # t = ((np.random.rand() * 2 - 1) * t, (np.random.rand() * 2 - 1) * t)
    # s = (np.random.rand() * 2 - 1) * s + 1
    # sh = (np.random.rand() * 2 - 1) * sh

    new_watermarked_image = transforms.functional.affine(watermarked_image, angle=r, translate=t, scale=s, shear=sh, fill=0)

    if Encoder is None:
        return new_watermarked_image

    with torch.no_grad():
        new_watermarked_latent = Encoder(new_watermarked_image, sample=False).detach()
        # ori_watermarked_latent = Encoder(watermarked_image, sample=False).detach()
        ori_watermarked_latent = None
    
    if init_latent is None:
        trans_init_latent = None
    else:
        t = (t[0]*init_latent.shape[-2]/watermarked_image.shape[-2], t[1]*init_latent.shape[-1]/watermarked_image.shape[-1])
        b, c, w, h = init_latent.shape
        trans_init_latent = transforms.functional.affine(init_latent.view(b*c, 1, w, h), angle=r, translate=t, scale=s, shear=sh, fill=0)
        trans_init_latent = trans_init_latent.view(b, c, w, h)
    
    return new_watermarked_image, new_watermarked_latent, ori_watermarked_latent, (r, t, s, sh), trans_init_latent


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img,seed, args):

    if args.jpeg_ratio is not None:
        r_id = random.randint(1, 99999)
        while os.path.exists(f"tmp_{r_id}.jpg"):
            r_id = random.randint(1, 99999)
        img.save(f"tmp_{r_id}.jpg", quality=args.jpeg_ratio)
        img = Image.open(f"tmp_{r_id}.jpg")
        os.remove(f"tmp_{r_id}.jpg")

    if args.random_crop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_crop_ratio)
        new_height = int(height * args.random_crop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        padded_image = np.zeros_like(img)
        padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
        img = Image.fromarray(padded_image)

    if args.random_drop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_drop_ratio)
        new_height = int(height * args.random_drop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
        img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
        img = Image.fromarray(img)

    if args.resize_ratio is not None:
        img_shape = np.array(img).shape
        resize_size = int(img_shape[0] * args.resize_ratio)
        img = transforms.Resize(size=resize_size)(img)
        img = transforms.Resize(size=img_shape[0])(img)

    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.median_blur_k is not None:
        img = img.filter(ImageFilter.MedianFilter(args.median_blur_k))


    if args.gaussian_std is not None:
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if args.sp_prob is not None:
        c,h,w = np.array(img).shape
        prob_zero = args.sp_prob / 2
        prob_one = 1 - prob_zero
        rdn = np.random.rand(c,h,w)
        img = np.where(rdn > prob_one, np.zeros_like(img), img)
        img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
        img = Image.fromarray(img)

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    return img


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)