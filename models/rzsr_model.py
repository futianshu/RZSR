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
