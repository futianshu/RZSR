为了严格复现《RZSR Randomly Initialized Zero-Shot Method for Blind Super-Resolution》中的工程，并满足你“构建整个项目工程”的要求，我为你设计了一个标准的 PyTorch 深度学习项目骨架。

在这个工程中：

1. **小论文中明确规定的细节**：例如**公式1**的随机退化模拟、$k=1000$ 和 $n=512$ 的知识字典参数、阶段一外部训练的 150 Epoch 及第 75 Epoch 学习率衰减、字典前 50 Epoch 不参与、内部 Zero-Shot 的 1-5 步微调且冻结字典、外部 `1e-4` 和内部 `1e-5` 的学习率等，都在代码中做了**硬性约束和严格实现**。
2. **小论文中未提及的内容**：例如具体的数据集加载逻辑、文件组织规范、以及论文中宣称可兼容任意 SR 网络的“基线残差模块”，我采用了业界规范进行了**工程化补充**，确保整个工程可以直接跑通。

---

### 📂 完整的项目工程目录结构

请在你的环境中创建以下目录结构：

```text
RZSR_Project/
├── requirements.txt            # 项目依赖
├── config.py                   # 全局超参数配置 (严格对齐论文参数)
├── data/
│   ├── dataset.py              # 外部大型数据集(如 DIV2K)的加载器
│   └── degradation.py          # 公式(1)的随机未知退化模拟引擎
├── models/
│   ├── aiem.py                 # 自适应信息提取模块 (AIEM/CondConv)
│   ├── knowledge_dict.py       # 知识字典模块 (KD)
│   ├── basic_sr_block.py       # 基础超分模块 (Model-Agnostic 占位)
│   └── rzsr_model.py           # 拼装 AIEM 与 KD 的最终网络骨架
├── train_external.py           # 阶段一：External Training
└── train_internal.py           # 阶段二：Internal Training (Zero-Shot) 及测试

```

下面是每一个文件的详细代码实现：

---

### 1. 核心依赖和配置

**`requirements.txt`**

```text
torch>=1.9.0
torchvision>=0.10.0
numpy
Pillow

```

**`config.py`**

```python
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

```

---

### 2. 数据处理与公式(1)退化模拟

**`data/degradation.py`**
严格实现论文**公式 (1)**：$I^{LR}=(I^{HR} * k)_{\downarrow s} + n$。

```python
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

```

**`data/dataset.py`**

```python
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ExternalDataset(Dataset):
    """ 用于阶段一真实 HR 图片加载 (如 DIV2K)，工程发挥补充 """
    def __init__(self, root_dir, patch_size=128):
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(img)

```

---

### 3. 网络架构与核心模块

**`models/aiem.py`**
严格遵循论文对 `condConv` 的描述（AvgPool -> 2 Convs -> Linear -> Softmax）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CondConvAttention(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_dim = max(1, in_channels // 4)
        self.conv1 = nn.Conv2d(in_channels, reduced_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(reduced_dim, in_channels, kernel_size=1)
        self.linear = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        b = x.size(0)
        out = self.avg_pool(x)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = out.view(b, -1)
        out = self.linear(out)
        return F.softmax(out, dim=1)

class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_experts=3):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.padding = kernel_size, padding
        self.num_experts = num_experts
        
        self.attention = CondConvAttention(in_channels, num_experts)
        self.weights = nn.Parameter(torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(num_experts, out_channels))
        nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.size()
        attn_weights = self.attention(x)
        
        weight_reshaped = self.weights.view(self.num_experts, -1)
        dyn_weight = torch.matmul(attn_weights, weight_reshaped)
        dyn_weight = dyn_weight.view(b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        dyn_bias = torch.matmul(attn_weights, self.bias).view(-1)
        
        # 利用 PyTorch 分组卷积 (groups) 的 Trick 实现 batch 内不同图片独立应用各自定制的卷积核
        x_reshaped = x.view(1, b * c, h, w)
        out = F.conv2d(x_reshaped, dyn_weight, bias=dyn_bias, padding=self.padding, groups=b)
        return out.view(b, self.out_channels, h, w)

class AIEM(nn.Module):
    """ 替换原始模型第一层卷积，包含数个 CondConv 层 """
    def __init__(self, in_channels, out_channels, num_experts=3):
        super().__init__()
        self.layer1 = CondConv2d(in_channels, out_channels, num_experts=num_experts)
        self.layer2 = CondConv2d(out_channels, out_channels, num_experts=num_experts)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))

```

**`models/knowledge_dict.py`**
严格遵循公式 $Q=\text{Inte}(\text{DR}(\text{feature}))$ 与矩阵注意力计算逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDictionary(nn.Module):
    def __init__(self, in_channels, k=1000, n=512):
        super().__init__()
        self.n_dim = n
        self.active = False # 控制前 50 epoch 是否隐去
        
        # 降维(DR) 与 插值映射为向量(Inte)
        self.dr = nn.Conv2d(in_channels, n, kernel_size=1)
        self.inte = nn.AdaptiveAvgPool2d(1) 
        
        # K, V 知识字典矩阵
        self.K = nn.Parameter(torch.randn(n, k))
        self.V = nn.Parameter(torch.randn(k, n))
        
        # 特征拼接混淆层
        self.confusion = nn.Conv2d(in_channels + n, in_channels, kernel_size=3, padding=1)
        
    def forward(self, feature):
        if not self.active:
            return feature
            
        b, c, h, w = feature.size()
        
        # Q = Inte(DR(feature))
        dr_feat = self.dr(feature)
        q = self.inte(dr_feat).view(b, self.n_dim) # [B, n]
        
        # Background Knowledge
        attn_scores = torch.matmul(q, self.K)             # [B, k]
        attn_weights = F.softmax(attn_scores, dim=-1)
        bg_knowledge = torch.matmul(attn_weights, self.V) # [B, n]
        
        # 空间分辨率恢复 (Interpolate and Reshape)
        bg_matrix = bg_knowledge.view(b, self.n_dim, 1, 1)
        bg_matrix = F.interpolate(bg_matrix, size=(h, w), mode='bilinear', align_corners=False)
        
        # 通道拼接与特征混淆
        concat_feat = torch.cat([feature, bg_matrix], dim=1)
        out = self.confusion(concat_feat)
        
        return out

```

**`models/basic_sr_block.py`**

```python
import torch.nn as nn

class BasicSRBlock(nn.Module):
    """ 基线 SR 网络特征提取占位符 (代指原论文中的 RDN / RCAN 等结构) """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

```

**`models/rzsr_model.py`**
按照论文 Figure 2 的拓扑结构，将 AIEM、SR Module、Knowledge Dictionary 包裹组装。

```python
import torch.nn as nn
from .aiem import AIEM
from .knowledge_dict import KnowledgeDictionary
from .basic_sr_block import BasicSRBlock

class RZSR_Model(nn.Module):
    def __init__(self, scale=2, in_channels=3, features=64, num_blocks=4, k=1000, n=512):
        super().__init__()
        # 1. 第一层卷积替换为 AIEM
        self.aiem = AIEM(in_channels, features)
        
        # 2. 每个基本 SR 模块后面跟着一个知识字典
        self.sr_modules = nn.ModuleList([BasicSRBlock(features) for _ in range(num_blocks)])
        self.kds = nn.ModuleList([KnowledgeDictionary(features, k=k, n=n) for _ in range(num_blocks)])
        
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.reconstruct = nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, x):
        x = self.aiem(x)
        for sr_block, kd in zip(self.sr_modules, self.kds):
            x = sr_block(x)
            x = kd(x)
        x = self.upsample(x)
        return self.reconstruct(x)

```

---

### 4. 阶段一：外部预训练脚本

**`train_external.py`**

```python
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

```

---

### 5. 阶段二：内部Zero-Shot微调及测试推理

**`train_internal.py`**

```python
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

```