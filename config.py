class Config:
    # --- SR 超分配置 ---
    SCALE = 2                # 放大倍数 (论文提及 2, 4)
    DEVICE = 'cuda'          # 运行设备
    
    # --- 退化配置 (Degradation IV.B) ---
    KERNEL_SIZE = 15         # 模糊核宽度 (The blur kernel width is 15)
    SIGMA_MIN = 0.2          # 论文 Table I 涵盖的模糊核分布
    SIGMA_MAX = 2.6
    
    # --- 知识字典配置 (KD III.C & IV.B) ---
    KD_K = 1000              # 字典向量数量 (k=1000)
    KD_N = 512               # 向量维度 (n=512)
    KD_START_EPOCH = 51      # 前 50 个 epoch 不引入 (not introduced in the first 50 epochs)
    
    # --- 阶段一：外部训练配置 (External Training) ---
    EXT_EPOCHS = 150         # 外部训练 150 个 epochs
    EXT_LR = 1e-4            # 初始化为 1e-4
    EXT_LR_DECAY_STEP = 75   # 跑到一半(75)时，学习率下降 10 倍
    EXT_BATCH_SIZE = 16
    
    # --- 阶段二：内部训练配置 (Internal Training) ---
    INT_STEPS = 5            # zero-shot 微调步数: 1-5 步
    INT_LR = 1e-5            # 内部训练学习率初始化为 1e-5
    INT_N_SONS = 10          # 为单张测试图随机初始化的退化版本数 N
