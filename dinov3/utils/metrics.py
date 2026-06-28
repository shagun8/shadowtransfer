"""
Evaluation metrics for shadow detection.
Implements metrics from the paper: OA, Precision, F1, BER, IOU.
"""

import torch
import numpy as np


class ShadowMetrics:
    """
    Evaluation metrics for shadow detection.
    
    Metrics (from paper, Equations 8-12):
    - Overall Accuracy (OA)
    - Precision (P)
    - F1 Score
    - Bit Error Rate (BER)
    - Intersection over Union (IOU)
    """
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.tp = 0  # True Positive
        self.tn = 0  # True Negative
        self.fp = 0  # False Positive
        self.fn = 0  # False Negative
        
    def update(self, pred, target):
        """
        Update metrics with batch predictions.
        
        Args:
            pred: Predictions [B, H, W] or [B, 2, H, W]
            target: Ground truth [B, H, W]
        """
        # Convert logits to predictions if needed
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        
        # Calculate confusion matrix elements
        # Shadow class is 1, non-shadow is 0
        self.tp += np.sum((pred == 1) & (target == 1))  # Shadow correctly detected
        self.tn += np.sum((pred == 0) & (target == 0))  # Non-shadow correctly detected
        self.fp += np.sum((pred == 1) & (target == 0))  # Non-shadow wrongly detected as shadow
        self.fn += np.sum((pred == 0) & (target == 1))  # Shadow wrongly detected as non-shadow
        
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        epsilon = 1e-7  # Avoid division by zero
        
        # Overall Accuracy (Equation 8)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + epsilon)
        
        # Precision (Equation 9)
        precision = self.tp / (self.tp + self.fp + epsilon)
        
        # F1 Score (Equation 10)
        f1 = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + epsilon)
        
        # Bit Error Rate (Equation 11)
        shadow_error = 1 - (self.tp / (self.tp + self.fn + epsilon))
        non_shadow_error = 1 - (self.tn / (self.tn + self.fp + epsilon))
        ber = 0.5 * (shadow_error + non_shadow_error)
        
        # Intersection over Union (Equation 12)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)

        # Shadow IoU (for shadow class = 1)
        shadow_iou = self.tp / (self.tp + self.fp + self.fn + epsilon)

        # Non-shadow IoU (for non-shadow class = 0)
        non_shadow_iou = self.tn / (self.tn + self.fp + self.fn + epsilon)

        # Mean IoU (average of both classes)
        miou = (shadow_iou + non_shadow_iou) / 2.0
        
        return {
            'OA': oa * 100,
            'Precision': precision * 100,
            'F1': f1 * 100,
            'BER': ber * 100,
            'Shadow_IOU': shadow_iou * 100,      # IoU for shadow class only
            'NonShadow_IOU': non_shadow_iou * 100,  # IoU for non-shadow class
            'mIOU': miou * 100,          # Average of both classes
            'IOU': shadow_iou * 100              # Keep original for backward compatibility
        }
    
    def __str__(self):
        """String representation of metrics"""
        metrics = self.compute()
        return (f"OA: {metrics['OA']:.2f}% | "
                f"Precision: {metrics['Precision']:.2f}% | "
                f"F1: {metrics['F1']:.2f}% | "
                f"BER: {metrics['BER']:.2f}% | "
                f"Shadow_IOU: {metrics['Shadow_IOU']:.2f}% | "
                f"mIOU: {metrics['mIOU']:.2f}%")


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: MAMNet model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = ShadowMetrics()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics.update(outputs, masks)
    
    return metrics.compute()


if __name__ == "__main__":
    # Test metrics
    metrics = ShadowMetrics()
    
    # Simulate some predictions
    pred = torch.randint(0, 2, (4, 256, 256))
    target = torch.randint(0, 2, (4, 256, 256))
    
    metrics.update(pred, target)
    
    print("Metrics:", metrics)
    print("\nDetailed:", metrics.compute())