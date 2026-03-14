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
