import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.degradation import apply_random_degradation
from data.dataset import ExternalDataset
from models.rzsr_model import RZSR_Model
from models.knowledge_dict import KnowledgeDictionary
from config import Config

def main():
    model = RZSR_Model(scale=Config.SCALE, k=Config.KD_K, n=Config.KD_N).to(Config.DEVICE)
    
    # 模拟数据加载
    train_dir = "./data/DIV2K_HR"
    if os.path.exists(train_dir):
        dataset = ExternalDataset(train_dir, patch_size=128)
        dataloader = DataLoader(dataset, batch_size=Config.EXT_BATCH_SIZE, shuffle=True)
    else:
        print(f"找不到 {train_dir}，使用 Dummy Data 假数据以验证工程逻辑跑通。")
        dataset = [torch.randn(3, 128, 128) for _ in range(32)]
        dataloader = DataLoader(dataset, batch_size=Config.EXT_BATCH_SIZE)

    criterion = nn.L1Loss()
    # 论文: LR 1e-4, drop at half epochs
    optimizer = optim.Adam(model.parameters(), lr=Config.EXT_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.EXT_LR_DECAY_STEP, gamma=0.1)

    model.train()
    print("=== 开始第一阶段：External Training ===")
    for epoch in range(1, Config.EXT_EPOCHS + 1):
        # 论文限制: knowledge dictionary is not introduced in the first 50 epochs
        kd_active = epoch >= Config.KD_START_EPOCH
        for module in model.modules():
            if isinstance(module, KnowledgeDictionary):
                module.active = kd_active
                
        epoch_loss = 0.0
        for hr_img in dataloader:
            hr_img = hr_img.to(Config.DEVICE)
            # 实时进行随机未知模糊降级
            with torch.no_grad():
                lr_img = apply_random_degradation(hr_img, Config.SCALE, Config.KERNEL_SIZE, Config.SIGMA_MIN, Config.SIGMA_MAX)
            
            optimizer.zero_grad()
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch [{epoch}/{Config.EXT_EPOCHS}], Loss: {epoch_loss/len(dataloader):.4f}, KD Active: {kd_active}")
        
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/rzsr_external.pth")
    print("External Training 完成！权重已保存。")

if __name__ == "__main__":
    main()
