import torch
import torch.nn as nn
import torch.nn.functional as F

class Weighted_LablesSmoothing_Loss(nn.modules.loss._WeightedLoss):
    def __init__(self, k=10, n_classes=5, alpha=0.1, gamma=3, weight=None, reduction='mean'):
        super(Weighted_LablesSmoothing_Loss, self).__init__(weight, reduction=reduction)
        if weight == None:
            weight = torch.tensor([1 for i in range(n_classes)])
        else:
            weight = weight / weight.sum()
        self.weight = weight
        self.alpha = alpha
        self.k = k
        self.gamma = gamma
        self.reduction = reduction
        self.n_classes = n_classes

    def forward(self, pred, true, cf=None):
        log_prob = F.log_softmax(pred, dim=-1)
        true_one_hot = F.one_hot(true, self.n_classes).float()
        true_one_hot = true_one_hot * (1 - self.alpha) + (self.alpha / self.n_classes) # Label smoothing
        weight = self.weight # Factor for manual weighting

        if cf == None:
            weight = self.weight.unsqueeze(0).repeat(true_one_hot.shape[0], 1).to(device=true_one_hot.device)
            loss = true_one_hot * -log_prob.to(device=log_prob.device) * weight
            loss = loss.sum(-1)

        else:
            cf = torch.tensor(cf)
            for i in range(len(cf)):
                if cf[i] <= 1e-4:
                    cf[i] = 1e-4
            w = torch.pow(1 -(torch.log(cf) / torch.log(torch.tensor(self.k))).float(), self.gamma).to(device=log_prob.device)
            w.require_grad = False
            loss = true_one_hot * -log_prob.to(device=log_prob.device) * weight * w
            loss = loss.sum(-1)

        if self.reduction == 'sum':
            loss = loss
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
#