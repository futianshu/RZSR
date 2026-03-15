import torch
import torch.nn as nn

class ESPCNBlock(nn.Module):
    """ 经典 ESPCN 浅层特征模块 """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)

class RCAB(nn.Module):
    """ RCAN 中的残差通道注意力模块 (Residual Channel Attention Block) """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x + self.body(x) * self.ca(self.body(x))
