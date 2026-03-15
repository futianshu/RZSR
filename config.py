import os
import torch

class Config:
    # --- SR 超参数与模型切换 ---
    SCALE = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 【重点发挥】你可以通过修改 BACKBONE，跑出不同的基准模型对比数据写进论文
    BACKBONE = 'RCAN'            # 可选: 'ESPCN', 'RCAN', 'RDN'
    NUM_BLOCKS = 4               
    FEATURES = 64                
    
    # --- 论文约束：退化环境 (IV.B 节) ---
    KERNEL_SIZE = 15             
    SIGMA_MIN, SIGMA_MAX = 0.2, 2.6  
    
    # --- 论文约束：KD 知识字典配置 ---
    KD_K = 1000                  
    KD_N = 512                   
    KD_START_EPOCH = 51          # 前 50 个 epoch 不引入字典
    
    # --- 论文约束：阶段一外部预训练 ---
    EXT_EPOCHS = 150             
    EXT_LR = 1e-4                
    EXT_LR_DECAY_STEP = 75       # 到 75 个 epoch 时学习率下降 10 倍
    BATCH_SIZE = 16
    PATCH_SIZE = 128             # 训练时的图片裁剪块大小
    
    # --- 论文约束：阶段二内部微调 ---
    INT_STEPS = 5                # 微调仅限 1-5 步
    INT_LR = 1e-5                
    INT_N_SONS = 10              # 随机初始化 N 个退化版本作为内部自监督标签
    
    # --- 工程路径配置 ---
    TRAIN_DIR = "./dataset/DIV2K_train_HR" 
    VAL_DIR = "./dataset/Set5"
    SAVE_DIR = "./checkpoints"
    LOG_DIR = f"./runs/RZSR_{BACKBONE}_X{SCALE}"

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)
