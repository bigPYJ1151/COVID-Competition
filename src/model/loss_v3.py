
import torch 
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight)
            weight.requires_grad = False
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, logit, label):
        '''
        inputs:
            logit
            label, non-Onehot
        '''
        # weight = self._class_weights(logit)
        if self.weight is not None:
            return F.cross_entropy(logit, label, weight=self.weight, ignore_index=self.ignore_index)
        else:
            return F.cross_entropy(logit, label, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = _Flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = nominator / denominator
        class_weights.requires_grad = False
        return class_weights

# class PixelWiseCrossEntropyLoss(nn.Module):
#     '''
#     Onehot function is wrong
#     '''

#     def __init__(self, class_weights=None, ignore_index=None):
#         super(PixelWiseCrossEntropyLoss, self).__init__()
#         self.register_buffer('class_weights', class_weights)
#         self.ignore_index = ignore_index
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, logit, label, weights):
#         '''
#         inputs:
#             logit
#             label, non-Onehot
#         '''
#         assert label.size() == weights.size()
#         # normalize the input
#         log_probabilities = self.log_softmax(logit)
#         # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
#         label = expand_as_one_hot(label, C=logit.size()[1], ignore_index=self.ignore_index)
#         # expand weights
#         weights = weights.unsqueeze(0)
#         weights = weights.expand_as(logit)

#         # create default class_weights if None
#         if self.class_weights is None:
#             class_weights = torch.ones(logit.size()[1]).float().to(logit.device)
#         else:
#             class_weights = self.class_weights

#         # resize class_weights to be broadcastable into the weights
#         class_weights = class_weights.view(1, -1, 1, 1, 1)

#         # multiply weights tensor by class weights
#         weights = class_weights * weights

#         # compute the losses
#         result = -weights * label * log_probabilities
#         # average the losses
#         return result.mean()

class _AbstractDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)

    def dice(self, prob, label, weight):
        raise NotImplementedError

    def forward(self, prob, label):
        # compute per channel Dice coefficient
        per_channel_dice = self.dice(prob, label, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    def __init__(self, epsilon=1e-6):
        super().__init__(weight=None)
        self.epsilon = epsilon

    def dice(self, prob, label, weight):
        assert prob.size() == label.size(), "'prob' and 'label' must have the same shape"

        prob = _Flatten(prob)
        label = _Flatten(label)
        label = label.float()

        if prob.size(0) == 1:
            prob = torch.cat((prob, 1 - prob), dim=0)
            label = torch.cat((label, 1 - label), dim=0)

        w_l = label.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (prob * label).sum(-1)
        intersect = intersect * w_l

        denominator = (prob + label).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

class DiceLoss(_AbstractDiceLoss):
    def __init__(self, weight=None):
        super().__init__(weight)

    def dice(self, prob, label, weight):
        return _ComputePreChannelDice(prob, label, weight=self.weight)

# class _MaskingLossWrapper(nn.Module):
#     '''
#     ToDo: Fix bugs to suit Dice Loss
#     '''
#     def __init__(self, loss, ignore_index):
#         super(_MaskingLossWrapper, self).__init__()
#         assert ignore_index is not None, 'ignore_index cannot be None'
#         self.loss = loss
#         self.ignore_index = ignore_index

#     def forward(self, input, target):
#         mask = target.clone().ne_(self.ignore_index)
#         mask.requires_grad = False

#         input = input * mask
#         target = target * mask

#         return self.loss(input, target)

def _ComputePreChannelDice(prob, label, epsilon=1e-6, weight=None):
    assert prob.size() == label.size(), "Size of prob and label mismatch!"

    prob = _Flatten(prob)
    label = _Flatten(label).float()

    intersect = (prob * label).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (prob * prob).sum(-1) + (label * label).sum(-1)
    
    return 2 * (intersect / denominator.clamp(min=epsilon))
    

def _Flatten(input_tensor):
    C = input_tensor.size(1)
    axis_order = (1,0) + tuple(range(2, input_tensor.dim()))
    
    transposed = input_tensor.permute(axis_order)

    return transposed.contiguous().view(C, -1)

# def expand_as_one_hot(input, C, ignore_index=None):
#     """
#     Todo: Maybe it is wrong...

#     Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
#     :param input: 4D input image (NxDxHxW)
#     :param C: number of channels/labels
#     :param ignore_index: ignore index to be kept during the expansion
#     :return: 5D output image (NxCxDxHxW)
#     """
#     assert input.dim() == 4

#     # expand the input tensor to Nx1xDxHxW before scattering
#     input = input.unsqueeze(1)
#     # create result tensor shape (NxCxDxHxW)
#     shape = list(input.size())
#     shape[1] = C

#     if ignore_index is not None:
#         # create ignore_index mask for the result
#         mask = input.expand(shape) == ignore_index
#         # clone the src tensor and zero out ignore_index in the input
#         input = input.clone()
#         input[input == ignore_index] = 0
#         # scatter to get the one-hot tensor
#         result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
#         # bring back the ignore_index in the result
#         result[mask] = ignore_index
#         return result
#     else:
#         # scatter to get the one-hot tensor
#         return torch.zeros(shape).to(input.device).scatter_(1, input, 1)