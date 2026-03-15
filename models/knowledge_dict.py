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
        
        # 彻底废除之前的 self.inte = nn.AdaptiveAvgPool2d(1)
        # 严格按照小论文的设定，在正向传播中使用插值来完成空间尺寸的压缩

        self.key = nn.Parameter(torch.randn(n_dim, k_dim))
        self.value = nn.Parameter(torch.randn(k_dim, n_dim))

        self.conv_out = nn.Conv2d(in_channels + n_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 执行降维 (DR)
        dr_x = self.dr(x)
        
        # 严格执行小论文的 Inte 操作来生成查询向量 Q
        # 利用插值将空间维度强制压缩到 1x1，随后展平
        q_inte = F.interpolate(dr_x, size=(1, 1), mode='bilinear', align_corners=False)
        q = q_inte.view(b, self.n_dim)

        # 计算相似度得分 Q * K
        qk = torch.matmul(q, self.key)
        attention = F.softmax(qk, dim=-1)

        # 严格对齐小论文带有转置的提取公式：V_bg = (Q * K)^T * V
        # 引入显式的转置操作 .t() 来满足数学表达式的要求
        attention_T = attention.t() 
        
        # 利用矩阵转置定律完成运算，保证输出形状依然是合法的 [b, n_dim]
        bg_knowledge = torch.matmul(self.value.t(), attention_T).t()

        # 将提取出的一维知识向量重塑为矩阵形状 (Reshape)
        bg_matrix = bg_knowledge.view(b, self.n_dim, 1, 1)
        
        # 再次利用插值 (Inte) 将其空间物理尺寸拉扯放大，恢复至与输入特征一致
        bg_matrix_up = F.interpolate(bg_matrix, size=(h, w), mode='bilinear', align_corners=False)

        # 将放大的外部知识矩阵与原始输入特征拼接，最后送入卷积层进行通道融合
        out = self.conv_out(torch.cat([x, bg_matrix_up], dim=1))

        return out