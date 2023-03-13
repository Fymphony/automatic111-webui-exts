import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# 从配置文件创建模型
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
    ddim_steps = 50
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
    
    # 提示词
    prompt = "a cute cat"
    # 批次 x 提示词 = [提示词,提示词 ...]
    data = [batch_size * [prompt]]

    # 一些杂事
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        # cfg scale
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        # 元组 转 列表
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        # c 就是embedding
                        c = model.get_learned_conditioning(prompts)
                        print("embedding shape:", c.shape)

                        # 采样数据 shape (4, 512/8, 512/8) => (4, 64, 64)
                        shape = [C, H // f, W // f]
                        # 采样
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        print("samples shape:", samples_ddim.shape)
                        # VAE 解码器 输出最终图像
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        # 把图像 值域 缩放到 0-1之间
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().numpy()

                        print("image shape:", x_samples_ddim.shape)
                        # 图像 值域从0-1 缩放到 0-255
                        x_sample = 255. * x_samples_ddim
                        # 保存成PNG文件
                        img1 = np.stack([x_sample[0][0,:,:], x_sample[0][1,:,:], x_sample[0][2,:,:]], axis=2)
                        img = Image.fromarray(img1.astype(np.uint8))
                        img.save(f"0.png")
     
if __name__ == "__main__":
    main()
