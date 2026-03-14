import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from models.rzsr_model import RZSR_Model
from models.knowledge_dict import KnowledgeDictionary
from data.degradation import apply_random_degradation
from config import Config
import copy

def main():
    base_model = RZSR_Model(scale=Config.SCALE, k=Config.KD_K, n=Config.KD_N).to(Config.DEVICE)
    if os.path.exists("checkpoints/rzsr_external.pth"):
        base_model.load_state_dict(torch.load("checkpoints/rzsr_external.pth", map_location=Config.DEVICE))
        print("已成功加载外部预训练权重。")
    
    # 拷贝模型，避免单张图片的自监督微调污染全局预训练模型
    model = copy.deepcopy(base_model)
    model.train()

    test_img_path = "./test_lr.png" # 现实中需要被超分的未知退化图片
    if os.path.exists(test_img_path):
        img = Image.open(test_img_path).convert('RGB')
        test_img_lr = transforms.ToTensor()(img).unsqueeze(0).to(Config.DEVICE)
    else:
        print(f"找不到测试图片 {test_img_path}，使用 Dummy Data 假数据以验证工程逻辑。")
        test_img_lr = torch.randn(1, 3, 64, 64).to(Config.DEVICE)

    # 1. 强制激活 KD 字典
    for module in model.modules():
        if isinstance(module, KnowledgeDictionary):
            module.active = True
            
    # 2. 论文核心限制：vectors in the KD do not participate in the update (对抗遗忘)
    for name, param in model.named_parameters():
        if 'K' in name or 'V' in name:
            param.requires_grad = False
            
    # 3. 内部微调学习率 initialized to 1e-5
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.INT_LR)
    criterion = nn.L1Loss()
    
    print(f"\n=== 开始内部 Zero-Shot 微调 (共 {Config.INT_STEPS} 步) ===")
    for step in range(1, Config.INT_STEPS + 1):
        optimizer.zero_grad()
        loss = 0
        
        # 将测试图当做 Label，为它获取 N 个劣化版本进行自监督
        for _ in range(Config.INT_N_SONS):
            lr_son = apply_random_degradation(test_img_lr, Config.SCALE, Config.KERNEL_SIZE, Config.SIGMA_MIN, Config.SIGMA_MAX)
            sr_son = model(lr_son)
            loss += criterion(sr_son, test_img_lr)
            
        loss = loss / Config.INT_N_SONS
        loss.backward()
        optimizer.step()
        print(f"Internal Step [{step}/{Config.INT_STEPS}], Semantic Loss: {loss.item():.5f}")

    # 微调结束后，重置冻结并切换为预测状态进行单图超分
    for name, param in model.named_parameters():
        if 'K' in name or 'V' in name:
            param.requires_grad = True
            
    model.eval()
    with torch.no_grad():
        final_sr = model(test_img_lr)
        
    final_img = transforms.ToPILImage()(final_sr.squeeze(0).cpu().clamp(0, 1))
    final_img.save("output_sr.png")
    print("\n✅ Zero-Shot Blind Super-Resolution 已完成。清晰图片结果已保存至 output_sr.png")

if __name__ == "__main__":
    main()
