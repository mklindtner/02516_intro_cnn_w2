import torch
import torch.nn.functional as F


def bce_loss(y_pred, y_target):
    return F.binary_cross_entropy(F.sigmoid(y_pred), y_target)


def dice_coefficient(pred, target):
    smooth = 1.
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def iou_loss(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return 1 - ((intersection + smooth) / (union + smooth))

def accuracy(pred, target):
    pred = torch.round(pred)
    correct = (pred == target).float()
    return correct.sum() / correct.numel()

def sensitivity(pred, target):
    true_positive = (pred * target).sum()
    possible_positive = target.sum()
    return true_positive / (possible_positive + 1e-6)

def specificity(pred, target):
    true_negative = ((1 - pred) * (1 - target)).sum()
    possible_negative = (1 - target).sum()
    return true_negative / (possible_negative + 1e-6)
