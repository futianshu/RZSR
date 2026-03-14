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
