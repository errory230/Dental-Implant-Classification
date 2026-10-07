"""ARPL loss (Chen et al., IEEE TPAMI 2021), following the official implementation
(https://github.com/iCGY96/ARPL, loss/ARPLoss.py and loss/Dist.py), one reciprocal point per class.

  d_e(x, P_k) = ||f(x) - P_k||^2 / D          (mean squared L2 distance)
  d_d(x, P_k) = f(x) . P_k                    (dot product)
  logit_k     = d_e(x, P_k) - d_d(x, P_k)     (larger = farther from reciprocal point k = more like class k)
  L_cls       = CE(logits / temp, y)
  L_o         = max(0, d_e(x, P_y) - R + 1)   (MarginRankingLoss(R, d_e, +1), learnable radius R)
  L           = L_cls + lambda * L_o
Prediction = argmax_k logit_k ; open-set score = max_k logit_k (higher = more likely known).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ARPLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, weight_pl=0.1, temp=1.0):
        super().__init__()
        self.weight_pl, self.temp = float(weight_pl), float(temp)
        self.points = nn.Parameter(0.1 * torch.randn(num_classes, feat_dim))
        self.radius = nn.Parameter(torch.zeros(1))
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def logits(self, x):
        d_l2 = (x.pow(2).sum(1, keepdim=True) - 2 * x @ self.points.t()
                + self.points.pow(2).sum(1).unsqueeze(0)) / x.shape[1]
        d_dot = x @ self.points.t()
        return d_l2 - d_dot

    def forward(self, x, labels=None):
        logits = self.logits(x)
        if labels is None:
            return logits, None
        loss_cls = F.cross_entropy(logits / self.temp, labels)
        d_known = (x - self.points[labels]).pow(2).mean(1)
        loss_r = self.margin_loss(self.radius.expand_as(d_known), d_known, torch.ones_like(d_known))
        return logits, loss_cls + self.weight_pl * loss_r
