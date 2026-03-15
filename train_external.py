import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.degradation import apply_random_degradation
from data.dataset import ExternalDataset
from models.rzsr_model import RZSR_Model
from models.knowledge_dict import KnowledgeDictionary
from config import Config
import copy

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)

def main():
    writer = SummaryWriter(log_dir=Config.LOG_DIR)
    model = RZSR_Model(backbone_type=Config.BACKBONE, scale=Config.SCALE, k=Config.KD_K, n=Config.KD_N).to(Config.DEVICE)
    
    # 初始化 EMA
    ema = ModelEMA(model)

    # 模拟数据加载
    train_dir = Config.TRAIN_DIR
    if os.path.exists(train_dir):
        dataset = ExternalDataset(train_dir, patch_size=Config.PATCH_SIZE)
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    else:
        print(f"找不到 {train_dir}，使用 Dummy Data 假数据以验证工程逻辑跑通。")
        dataset = [torch.randn(3, Config.PATCH_SIZE, Config.PATCH_SIZE) for _ in range(32)]
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)

    criterion = nn.L1Loss()
    # 论文: LR 1e-4, drop at half epochs
    optimizer = optim.Adam(model.parameters(), lr=Config.EXT_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.EXT_LR_DECAY_STEP, gamma=0.1)

    model.train()
    print(f"=== 开始第一阶段：External Training ({Config.BACKBONE}) ===")
    for epoch in range(1, Config.EXT_EPOCHS + 1):
        # 论文限制: knowledge dictionary is not introduced in the first 50 epochs
        kd_active = epoch >= Config.KD_START_EPOCH
        for module in model.modules():
            if isinstance(module, KnowledgeDictionary):
                module.active = kd_active
                
        epoch_loss = 0.0
        for step, hr_img in enumerate(dataloader):
            hr_img = hr_img.to(Config.DEVICE)
            # 实时进行随机未知模糊降级
            with torch.no_grad():
                lr_img = apply_random_degradation(hr_img, Config.SCALE, Config.KERNEL_SIZE, Config.SIGMA_MIN, Config.SIGMA_MAX)
            
            optimizer.zero_grad()
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            loss.backward()
            optimizer.step()
            
            # 更新 EMA
            ema.update(model)
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(dataloader)
        scheduler.step()
        print(f"Epoch [{epoch}/{Config.EXT_EPOCHS}], Loss: {avg_loss:.4f}, KD Active: {kd_active}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # 保存 EMA 模型而不是原始模型
    torch.save(ema.ema_model.state_dict(), os.path.join(Config.SAVE_DIR, "rzsr_best.pth"))
    print("External Training 完成！EMA 平滑权重已保存至 rzsr_best.pth。")
    writer.close()

if __name__ == "__main__":
    main()
