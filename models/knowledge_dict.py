import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDict(nn.Module):
    def __init__(self, in_channels, n_dim=512, k_dim=1000):
        super(KnowledgeDict, self).__init__()
        self.n_dim = n_dim
        self.k_dim = k_dim

        # DR (Dimensionality Reduction) 降维操作
        self.dr = nn.Conv2d(in_channels, n_dim, kernel_size=1)
        
        # 【修改点 1：删除原有的池化操作】
        # 注释掉或删除原来这行：self.inte = nn.AdaptiveAvgPool2d(1)

        self.key = nn.Parameter(torch.randn(n_dim, k_dim))
        self.value = nn.Parameter(torch.randn(k_dim, n_dim))

        self.conv_out = nn.Conv2d(in_channels + n_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 执行降维
        dr_x = self.dr(x)
        
        # 【修改点 1 对应执行：严格使用插值 (Interpolation) 代替池化生成 Q】
        # 利用 F.interpolate 将空间维度强行插值压缩到 1x1，并展平为 [b, n_dim]
        q = F.interpolate(dr_x, size=(1, 1), mode='bilinear', align_corners=False).view(b, self.n_dim)

        # 计算 Q * K
        qk = torch.matmul(q, self.key)
        attention = F.softmax(qk, dim=-1)

        # 【修改点 2：严格对齐小论文带有转置的公式 V_bg = (Q * K)^T * V】
        # 显式引入转置操作 .t()，并通过矩阵转置定律保证 Batch 维度运算合法
        attention_T = attention.t()  # 这一步严格对应公式中的 (Q*K)^T，形状变为 [k_dim, b]
        
        # (V^T * (Q*K)^T)^T 在数学上等价于 (Q*K) * V
        # 这样写既在代码物理层面上执行了公式要求的转置，又保证了输出形状被正确还原为 [b, n_dim]
        bg_knowledge = torch.matmul(self.value.t(), attention_T).t()

        # 将提取的知识向量重塑形状
        bg_matrix = bg_knowledge.view(b, self.n_dim, 1, 1)
        
        # 再次利用插值将其放大回与输入特征相同的物理尺寸
        bg_matrix_up = F.interpolate(bg_matrix, size=(h, w), mode='bilinear', align_corners=False)

        # 将外部知识与原始特征拼接后进行卷积融合
        out = self.conv_out(torch.cat([x, bg_matrix_up], dim=1))

        return out