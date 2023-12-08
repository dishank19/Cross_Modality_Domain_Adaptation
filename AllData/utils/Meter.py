import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from pytorch_msssim import ssim

def dice_coef_metric(predictions: torch.Tensor, 
                     truth: torch.Tensor,
                     num_classes = 1,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice Score for data batch.
    Params:
        predictions: predicted classes (0, 1, 2, ...).
        truth: ground truth classes.
        eps: additive to refine the estimate.
        Returns: dice score.
    """
    if num_classes == 1:
        intersection = 2.0 * (predictions * truth).sum(axis = (1, 2, 3))
        union = predictions.sum(axis = (1, 2, 3)) + truth.sum(axis = (1, 2, 3))
        intersection = intersection + eps
        union = union + eps
        return [(intersection / union).mean()]
    
    total_dice = []
    for i in range(num_classes):
        pred = (predictions == i).float()
        true = (truth == i).float()
        
        intersection = 2.0 * (true * pred).sum(axis = (1, 2))
        union = true.sum(axis = (1, 2)) + pred.sum(axis = (1, 2))
        
        intersection = intersection + eps
        union = union + eps
        
        dice_score = (intersection / union).mean()
            
        total_dice.append(dice_score)

    return total_dice


def jaccard_coef_metric(predictions: torch.Tensor,
                        truth: torch.Tensor,
                        num_classes = 1,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        predictions: predicted classes (0, 1, 2, ...).
        truth: ground truth classes.
        eps: additive to refine the estimate.
        Returns: jaccard score.
    """
    
    if num_classes == 1:
        intersection = (predictions * truth).sum(axis = (1, 2, 3))
        union = (predictions + truth).sum(axis = (1, 2, 3)) - intersection
        
        intersection = intersection + eps
        union = union + eps

        return [(intersection / union).mean()]
    
    total_jaccard = []
    for i in range(num_classes):
        pred = (predictions == i).float()
        true = (truth == i).float()
        
        intersection = (pred * true).sum(axis = (1, 2))
        union = (pred + true).sum(axis = (1, 2)) - intersection
        
        intersection = intersection + eps
        union = union + eps
        
        jaccard_score = (intersection / union).mean()
        
        total_jaccard.append(jaccard_score)

    return total_jaccard


class Meter:
    '''Factory for storing and updating iou and dice scores.'''

    def __init__(self):
        self.dice_scores: list = []
        self.iou_scores: list = []
        self.num_classes = None  # Initially set as None

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        # Infer number of classes if it's the first call
        if self.num_classes is None:
            self.num_classes = logits.shape[1]

        if self.num_classes == 1:  # Binary segmentation
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
        else:  # Multi-class segmentation
            targets = targets.squeeze(1)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

        dice = dice_coef_metric(predictions, targets, self.num_classes)
        iou = jaccard_coef_metric(predictions, targets, self.num_classes)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(np.array(self.dice_scores))
        iou = np.mean(np.array(self.iou_scores))
        return dice, iou
    
    def get_metrics_by_class(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(np.array(self.dice_scores), axis = 0)
        iou = np.mean(np.array(self.iou_scores), axis = 0)
        return dice, iou
    

class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)

        intersection = 2.0 * (probability * targets).sum(axis = 1)
        union = probability.sum(axis = 1) + targets.sum(axis = 1)
        dice_score = (intersection + self.eps) / (union + self.eps)
        return 1.0 - dice_score.mean()
    
    
class BCELoss(nn.Module):
    """Compute objective loss: BCE loss."""

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        return bce_loss
    

class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)

        return bce_loss + dice_loss
    
    
class MultiClassDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super(MultiClassDiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        probabilities = torch.softmax(logits, dim=1)
    
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=probabilities.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)

        assert (probabilities.shape == targets_one_hot.shape)

        intersection = torch.sum(probabilities * targets_one_hot, dim=(2, 3)) * 2.0
        union = torch.sum(probabilities, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        
        intersection = intersection + self.eps
        union = union + self.eps
        
        dice_scores = intersection / union

        return 1.0 - torch.mean(torch.mean(dice_scores, axis = 0))
    

class CrossEntropyDiceLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        targets = targets.squeeze(1)
        
        assert (logits.shape[1] > 1)
        assert (logits.shape[0] == targets.shape[0])

        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())

        return ce_loss + dice_loss