import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class temporal_crossentropy_loss(nn.modules.loss._WeightedLoss):
    def __init__(self):
        super(temporal_crossentropy_loss, self).__init__()

    def forward(self, output, target):
        return F.cross_entropy(output, target, reduction='mean')

# class Weighted_LablesSmoothing_Loss(nn.modules.loss._WeightedLoss):
#     def __init__(self, k=100, n_classes=5, alpha=0.1, weight=None, reduction='mean'):
#         super(Weighted_LablesSmoothing_Loss, self).__init__(weight, reduction=reduction)
#         self.k = k
#         self.weight = weight
#         self.alpha = alpha
#         self.reduction = reduction
#         self.n_classes = n_classes
#
#     def forward(self, pred, true, cf=None):
#         weight = self.weight
#         if cf == None:
#             loss = F.cross_entropy(pred, true, weight=weight)
#
#         else:
#             with torch.no_grad():
#                 arg_pred = list(F.softmax(pred, dim=-1).argmax(-1).cpu().numpy())
#                 arg_true = list(true.cpu().numpy())
#             idx_pairs = list(zip(arg_true, arg_pred))
#
#             w_list = []
#             for idx in idx_pairs:
#                 # if cf[idx] == 1:
#                 if cf[idx] == 0:
#                     cf[idx] = 1e-5
#
#                 # if idx[0] == idx[1] or -math.log(self.k, cf[idx]) <= 1:
#                 if idx[0] == idx[1] or self.k * cf[idx] >= 1:
#                     w_list.append(1)
#                 else:
#                     # w_list.append(-math.log(self.k, cf[idx]))
#                     w_list.append(self.k * cf[idx])
#             loss = F.cross_entropy(pred, true, weight=weight) * torch.tensor(w_list).to(device=true.device)
#
#         if self.reduction == 'sum':
#             loss = loss
#         elif self.reduction == 'mean':
#             loss = loss.mean()
#         return loss

class Weighted_LablesSmoothing_Loss(nn.modules.loss._WeightedLoss):
    def __init__(self, k=2, n_classes=5, alpha=0, gamma=3, weight=None, reduction='mean'):
        super(Weighted_LablesSmoothing_Loss, self).__init__(weight, reduction=reduction)
        if weight == None:
            weight = torch.tensor([1 for i in range(n_classes)])
        else:
            weight = weight / weight.sum()
        self.k = k
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.n_classes = n_classes

    def forward(self, pred, true, cf=None):
        log_prob = F.log_softmax(pred, dim=-1)
        true_one_hot = F.one_hot(true, self.n_classes).float()
        true_one_hot = true_one_hot * (1 - self.alpha) + (self.alpha / self.n_classes) ## Label smoothing
        weight =self.weight

        if cf == None:
            # loss = F.cross_entropy(pred, true, weight=weight, reduction=self.reduction)
            weight = self.weight.unsqueeze(0).repeat(true_one_hot.shape[0], 1).to(device=true_one_hot.device)
            loss = true_one_hot * -log_prob.to(device=log_prob.device) * weight
            loss = loss.sum(-1)

        else:
            # raise 'error'
            cf = torch.tensor(cf)
            for i in range(len(cf)):
                if cf[i] <= 1e-4:
                    cf[i] = 1e-4

            # cf = torch.pow(1+(1-torch.tensor(cf)), 3).to(device=pred.device)/
            cf = torch.pow(1 -(torch.log(cf) / torch.log(torch.tensor(self.k))).float(), self.gamma).to(device=log_prob.device)
            # for i in range(len(cf)):
            #     for j in range(len(cf[i])):
            #         if i != j:
            #             cf[i][j] = 0
            # cf = F.normalize(cf, dim=-1)
            # print(cf)
            # cf = cf.sum(-1).float()
            cf.require_grad = False

            # loss = F.cross_entropy(pred, true, weight=cf, reduction=self.reduction)
            loss = true_one_hot * -log_prob.to(device=log_prob.device) * weight * cf
            # loss = loss.sum(-1)
            # loss = torch.matmul((true_one_hot * -log_prob), cf).to(device=log_prob.device)

        if self.reduction == 'sum':
            loss = loss
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
#