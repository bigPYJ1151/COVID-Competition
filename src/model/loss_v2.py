
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 

class CAMConstrain(nn.Module):

    def __init__(self, weight=1):
        super().__init__()

        self.weight = weight

    def forward(self, cam, camlabel):
        cam = torch.abs(cam)
        camlabel = F.one_hot(camlabel).movedim(4, 1)[:, 0]
        camlabel = torch.unsqueeze(camlabel, dim=1)
        camlabel = 1 - camlabel
        
        N, C, D, H, W = cam.size()
        camlabel = camlabel.expand_as(cam).view(N, C, -1)
        cam = cam.view(N, C, -1)

        upper = cam * camlabel

        upperPercent = upper.sum(dim=2) / cam.sum(dim=2) 
        loss = -torch.log(upperPercent)

        return self.weight * loss.mean(dim=1)

class CAMConstrainDirection(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()

        self.weight = weight

    def forward(self, cam, camlabel, label):
        camlabel = F.one_hot(camlabel).movedim(4, 1)[:, 0]
        camlabel = torch.unsqueeze(camlabel, dim=1)
        
        N, C, D, H, W = cam.size()
        camlabel = camlabel.expand_as(cam).view(N, C, -1)
        cam = cam.view(N, C, -1)

        # upper = (cam * (1 - camlabel) / (camlabel.sum(dim=2, keepdim=True))).sum(dim=2, keepdim=True)
        upper = (cam * (1 - camlabel)).mean(dim=2, keepdim=True)
  
        upper = torch.softmax(upper, dim=2)
        upper = -torch.log(upper)

        loss = upper[F.one_hot(label, 2).to(torch.bool)]

        return self.weight * loss

class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5):
        '''
        alpha is the weight of precision
        beta is the weight of recall
        '''
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, prob, label):
        N, C, D, H, W = prob.size()

        prob = prob.view(N, C, -1)
        label = label.view(N, C, -1)

        tp = (prob * label).sum(dim=2)
        fp = (prob * (1 - label)).sum(dim=2)
        fn = ((1 - prob) * label).sum(dim=2)

        loss = tp[:, 1:].sum(dim=1) / (tp + self.alpha * fp + self.beta * fn)[:, 1:].sum(dim=1)

        return 1 - loss

class TverskyGDiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, prob, label):
        N, C, D, H, W = prob.size()
        
        prob_mask = prob.argmax(dim=1)
        prob_mask = F.one_hot(prob_mask, C).type_as(label).movedim(4, 1).view(N, C, -1)

        union = (prob * label).view(N, C, -1)
        prob_r = (prob).view(N, C, -1)
        label_r = label.view(N, C, -1)

        TP_mask = (prob_mask * label_r).detach()
        FP_mask = (prob_mask * (1 - label_r) * 2 * self.alpha).detach()
        FN_mask = ((1 - prob_mask) * label_r * 2 * self.beta).detach()
        final_mask = TP_mask + FP_mask + FN_mask


        weight = 1 / ((label.view(N, C, -1).sum(dim=2)) ** 2 + 1)

        loss = ((union.sum(dim=2) * weight)[:, 1:].sum(dim=1)) / ((((prob_r * final_mask.detach()).sum(dim=2) + (label_r * final_mask.detach()).sum(dim=2)) * weight)[:, 1:].sum(dim=1)) 
        loss = 1 - 2 * loss

        return loss 

class GDiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, prob, label):
        N, C, D, H, W = prob.size()

        union = (prob * label).view(N, C, -1).sum(dim=2)
        prob_r = (prob ** 2).view(N, C, -1).sum(dim=2)
        label_r = label.view(N, C, -1).sum(dim=2)
        weight = 1 / ((label.view(N, C, -1).sum(dim=2)) ** 2 + 1)

        loss = ((union * weight)[:, 1:].sum(dim=1)) / (((prob_r + label_r) * weight)[:, 1:].sum(dim=1)) 
        loss = 1 - 2 * loss

        return loss

class GDiceLoss_v2(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, prob, label):
        N, C, D, H, W = prob.size()

        union = (prob * label).view(N, C, -1).sum(dim=2)
        prob_r = prob.view(N, C, -1).sum(dim=2)
        label_r = label.view(N, C, -1).sum(dim=2)
        weight = 1 / ((label.view(N, C, -1).sum(dim=2)) ** 2 + 1)

        loss = ((union * weight)[:, 1:].sum(dim=1)) / (((prob_r + label_r) * weight)[:, 1:].sum(dim=1)) 
        loss = 1 - 2 * loss

        return loss

class CELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logit, label):
        N, C, D, H, W = logit.size()
        
        label_max = label.argmax(dim=1)

        loss = F.cross_entropy(logit, label_max, reduction='none')
        loss = loss.unsqueeze(1).expand_as(label)
        loss = loss * label.type_as(loss)
        loss = loss.view(N, C, -1).sum(dim=2)
        num_positive = label.view(N, C, -1).sum(dim=2).type_as(loss)
        loss = loss / (num_positive + 1)
        loss = loss[:, :].mean(dim=1)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit, label):
        N, C, D, H, W = logit.size()
        
        prob = F.softmax(logit, dim=1) * label.type_as(logit)
        prob = prob.sum(dim=1)
        label_max = label.argmax(dim=1)
        
        loss = F.cross_entropy(logit, label_max, reduction='none')

        F_loss = self.alpha * (1 - prob)**self.gamma * loss

        F_loss = F_loss.unsqueeze(1).expand_as(label)
        F_loss = F_loss * label.type_as(F_loss)
        F_loss = F_loss.view(N, C, -1).sum(dim=2)
        num_positive = label.view(N, C, -1).sum(dim=2).type_as(F_loss)
        F_loss = F_loss / (num_positive + 1)
        F_loss = F_loss[:, :].mean(dim=1)

        return F_loss
        
