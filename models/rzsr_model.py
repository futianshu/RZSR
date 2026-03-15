import torch.nn as nn
from .aiem import AIEM
from .knowledge_dict import KnowledgeDictionary
from .backbones import ESPCNBlock, RCAB
from config import Config  # Import Config for ablation flags

class RZSR_Model(nn.Module):
    def __init__(self, backbone_type='RCAN', scale=2, in_channels=3, features=64, num_blocks=4, k=1000, n=512):
        super().__init__()
        
        # 1. AIEM 消融支持
        if getattr(Config, 'USE_AIEM', True):
            self.aiem = AIEM(in_channels, features, num_experts=3)
        else:
            self.aiem = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding=1)
            )
        
        # 挂载指定的基准网络底座与知识字典
        self.sr_modules = nn.ModuleList()
        self.kds = nn.ModuleList()
        for _ in range(num_blocks):
            if backbone_type == 'ESPCN': self.sr_modules.append(ESPCNBlock(features))
            elif backbone_type == 'RCAN': self.sr_modules.append(RCAB(features))
            else: raise ValueError("未支持的基准模型")
            
            # 2. KD 知识字典消融支持
            if getattr(Config, 'USE_KD', True):
                self.kds.append(KnowledgeDictionary(features, k=k, n=n))
            else:
                self.kds.append(nn.Identity()) # 关闭 KD 时，使用透明的恒等映射占位
        
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
