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
