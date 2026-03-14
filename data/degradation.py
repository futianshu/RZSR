import torch
import torch.nn.functional as F
import random

def apply_random_degradation(img_hr, scale, kernel_size, sigma_min, sigma_max):
    """ 施加随机退化以模拟未知退化环境 """
    b, c, h, w = img_hr.shape
    device = img_hr.device
    
    # 随机初始化高斯模糊核参数
    sigma = random.uniform(sigma_min, sigma_max)
    
    # 1. 生成随机高斯模糊核 k
    grid = torch.arange(kernel_size).float() - kernel_size // 2
    x_grid = grid.view(1, -1).to(device)
    y_grid = grid.view(-1, 1).to(device)
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    
    # 应用模糊 (* k) -> 使用反射填充防止边缘伪影
    pad = kernel_size // 2
    img_pad = F.pad(img_hr, (pad, pad, pad, pad), mode='reflect')
    img_blur = F.conv2d(img_pad, kernel, groups=c)
    
    # 2. 下采样 (\downarrow s)
    img_lr = F.interpolate(img_blur, scale_factor=1.0/scale, mode='bicubic', align_corners=False)
    
    # 3. 添加高斯白噪声 (+ n)
    noise_level = random.uniform(0, 5) / 255.0
    noise = torch.randn_like(img_lr) * noise_level
    img_lr = img_lr + noise
    
    return torch.clamp(img_lr, 0.0, 1.0)
