import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import Config
from data.degradation import apply_random_degradation
from models.rzsr_model import RZSR_Model
from models.knowledge_dict import KnowledgeDictionary
from utils.metrics import calc_psnr

def internal_zero_shot_adaptation(base_model, test_img_lr, scale):
    """ 对单图进行内部 Zero-Shot 微调 (严格执行论文约束) """
    model = copy.deepcopy(base_model)
    model.train()
    
    # 约束1：强制激活 KD 字典
    for module in model.modules():
        if isinstance(module, KnowledgeDictionary): module.active = True
            
    # 约束2：对抗灾难性遗忘，完全冻结字典向量的梯度更新 (写论文核心讨论点)
    for name, param in model.named_parameters():
        if 'K' in name or 'V' in name: param.requires_grad = False
            
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.INT_LR)
    criterion = torch.nn.L1Loss()
    
    # 构造 N 个由 LR 进一步退化的自监督儿子数据
    son_pairs = []
    with torch.no_grad():
        for _ in range(Config.INT_N_SONS):
            son_pairs.append(apply_random_degradation(test_img_lr, scale, Config.KERNEL_SIZE, Config.SIGMA_MIN, Config.SIGMA_MAX))
            
    # 仅极速微调 1-5 步
    for step in range(Config.INT_STEPS):
        optimizer.zero_grad()
        loss = 0
        for lr_son in son_pairs:
            sr_son = model(lr_son)
            loss += criterion(sr_son, test_img_lr) # 语义一致性约束
        loss = loss / Config.INT_N_SONS
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad(): return model(test_img_lr)

def main():
    print(f"========== 启动论文 Table I / II 评估管线 | 底座: {Config.BACKBONE} ==========")
    base_model = RZSR_Model(backbone_type=Config.BACKBONE, scale=Config.SCALE).to(Config.DEVICE)
    
    # 这里模拟从第一阶段 train_external.py 跑完保存下来的最优权重
    weight_path = os.path.join(Config.SAVE_DIR, "rzsr_best.pth")
    if os.path.exists(weight_path):
        base_model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
    else:
        print("⚠️ 警告：未找到预训练权重，为了走通工程，当前使用随机初始化参数。")

    # 如果你本地没有测试集，这里会自动构造 Dummy 数据以保证代码能跑通
    try:
        from data.dataset import SRDataset
        test_loader = DataLoader(SRDataset(Config.VAL_DIR, is_train=False), batch_size=1)
    except:
        test_loader = [(torch.rand(1, 3, 256, 256), "dummy.png")] * 5

    out_visual_dir = "./results_visual"
    os.makedirs(out_visual_dir, exist_ok=True)
    sigmas_to_test = [0.0, 0.2, 1.3, 2.6]  # 对应小论文中的测试核大小
    
    for sigma in sigmas_to_test:
        total_psnr = 0.0
        print(f"\n=> 正在评测环境 Kernel Sigma = {sigma} ...")
        
        for idx, batch in enumerate(test_loader):
            hr_img = batch[0].to(Config.DEVICE) if isinstance(batch, tuple) else batch.to(Config.DEVICE)
            
            with torch.no_grad():
                # 模拟测试环境下的低清图像
                lr_img = apply_random_degradation(hr_img, Config.SCALE, Config.KERNEL_SIZE, Config.SIGMA_MIN, Config.SIGMA_MAX, fixed_sigma=sigma)
            
            # 【执行推理】内部零样本学习
            sr_img = internal_zero_shot_adaptation(base_model, lr_img, Config.SCALE)
            
            psnr = calc_psnr(sr_img, hr_img, crop_border=Config.SCALE)
            total_psnr += psnr
            
            # 【毕业论文神技】自动拼接：[Bicubic 插值 | RZSR 超分 | HR 原图]
            if idx == 0:  # 仅保存每组环境的第一张图作为论文样例展示
                bicubic = torch.nn.functional.interpolate(lr_img, scale_factor=Config.SCALE, mode='bicubic')
                comp_img = torch.cat([bicubic.squeeze(0), sr_img.squeeze(0), hr_img.squeeze(0)], dim=2).cpu().clamp(0, 1)
                from torchvision.transforms import ToPILImage
                ToPILImage()(comp_img).save(os.path.join(out_visual_dir, f"Sigma_{sigma}_compare.png"))
            
        avg_psnr = total_psnr / len(test_loader)
        print(f"Sigma={sigma} | 平均 PSNR: {avg_psnr:.2f} dB (已导出视觉对比图)")

if __name__ == "__main__":
    main()
