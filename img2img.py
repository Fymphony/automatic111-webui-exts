"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
      # 设置随机数种子
    seed_everything(42)
    
    # 加载配置文件
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")

   
    # 从加载的配置文件创建模型
    model = load_model_from_config(config, "models\ldm\stable-diffusion-v1\model.ckpt")

    # 判断GPU或者CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 将模型送到CUDA设备 或者 CPU
    model = model.to(device)

    # 创建ddim 采样器
    sampler = DDIMSampler(model)

    # 批次大小
    batch_size = 1
    # 起始编码 (没用)
    start_code = None
    # CFG 缩放
    scale = 7.5
    # 采样步数
    ddim_steps = 100
    # 通道数
    C = 4
    # 高
    H = 512
    # 宽
    W = 512
    # 下采样因子
    f = 8
    # 采样数量
    n_samples = 1
    # 确定性因子
    ddim_eta = 0.0
    # 迭代次数
    n_iter = 1
    # 去除噪声强度
    strength = 0.75
    
    # 提示词
    prompt = "a cute cat"
    # 批次 x 提示词 = [提示词,提示词 ...]
    data = [batch_size * [prompt]]
  
    # 加载 像素空间 图片
    init_image = load_img("0.png").to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

    # 像素空间 转换到 潜在空间
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    # 初始化采样器
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    # 取噪强度 * 采样步数
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast 
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent) Aplha混合 潜在空间图像 和 噪声
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it 采样器采样
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,)

                        # VAE 解码器 将潜在空间转化为像素空间
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = x_samples.cpu().numpy()

                        print("image shape:", x_samples.shape)
                        # 图像 值域从0-1 缩放到 0-255
                        x_sample = 255. * x_samples
                        
                        # 保存成PNG文件
                        img1 = np.stack([x_sample[0][0,:,:], x_sample[0][1,:,:], x_sample[0][2,:,:]], axis=2)
                        img = Image.fromarray(img1.astype(np.uint8))
                        img.save(f"1.png")

if __name__ == "__main__":
    main()
