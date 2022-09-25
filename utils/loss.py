from re import I
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MySoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self, nbclasses):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, self.nbclasses)  # N,H*W,C => N*H*W,C
        target = target.view(-1) # N*H*W
        return nn.CrossEntropyLoss(reduction="mean")(inputs, target)

        # N*H*W,C
        # N*H*W

        # 2*3*3, 3
        # 2*3*3


        # bce
        # 2*3*3
        # 2*3*3


class SoftDiceLoss(nn.Module):
    def __init__(self, nbclasses, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.nbclasses = nbclasses

    def one_hot(self, targets):
        n, h, w = targets.size(0), targets.size(1), targets.size(2) # n h w
        targets = targets.reshape(-1) # n*h*w
        targets = torch.eye(self.nbclasses)[targets] # n*h*w c
        targets = targets.view(n, h*w, -1) # n h*w c
        return targets
 
    def forward(self, logits, targets):
        # logits    n c h w
        # targets   n h w
        n, c, h, w = logits.size(0), logits.size(1), logits.size(2), logits.size(3)
        targets = self.one_hot(targets) # n h*w c
        smooth = 1
        
        logits = logits.permute(0, 2, 3, 1)# n h w c
        logits = logits.contiguous().view(n, -1, c)# n h*w c


        logc = (-torch.max(logits, 2)[0]).reshape(n, h*w, 1) # n h*w 1
        logits = logc + logits

        soft_logits = torch.exp(logits)/torch.sum(torch.exp(logits), 2).reshape(n, h*w, -1) # n h*w, c


        m1 = soft_logits.view(n, -1).cuda() # n h*w*c
        m2 = targets.view(n, -1).cuda() # n h*w*c
        intersection = (m1 * m2) # n h*w*c
 
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth) # intersection.sum(1) n || score n tensor([3., 4.]) torch.Size([2])
        score = 1 - score.sum() / n # score.sum() tensor(7.) torch.Size([])
        return score
