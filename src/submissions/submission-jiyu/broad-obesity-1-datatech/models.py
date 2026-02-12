import torch
import torch.nn as nn


class CosineRidge(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=10.0):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.alpha = alpha  # L2 正则系数

    def forward(self, x):
        return self.W(x)

    def loss(self, y_pred, y_true):
        # 中心化（这是关键！对齐 Pearson）
        y_pred_c = y_pred - y_pred.mean(dim=1, keepdim=True)
        y_true_c = y_true - y_true.mean(dim=1, keepdim=True)

        cos_sim = nn.functional.cosine_similarity(y_pred_c, y_true_c, dim=1)
        cosine_loss = 1.0 - cos_sim.mean()

        # L2 正则（Ridge）
        l2 = torch.sum(self.W.weight ** 2)
        return cosine_loss + self.alpha * l2
