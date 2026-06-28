"""
Post-processing utilities for shadow detection predictions.

Key fix: filter_small_predictions now uses argmax to determine the predicted
class, NOT a raw-logit threshold of 0.5.

The threshold bug:
    pred_binary = (predictions[i, 1] > 0.5)   ← WRONG: depends on logit magnitude
    pred_binary = argmax(predictions[i]) == 1  ← CORRECT: depends only on relative ordering

Logit magnitudes are irrelevant for classification — argmax and softmax both
depend only on relative ordering. However, a hard 0.5 threshold on raw logits
will silently discard all shadow predictions whenever logit magnitudes are
suppressed (e.g. by ISW regularization), causing mIOU = 0 despite CE loss
decreasing normally.
"""

import torch
import numpy as np
import cv2


def filter_small_predictions(predictions, min_pixels=10):
    """
    Remove small connected components from shadow predictions.

    Args:
        predictions: Tensor [B, 2, H, W]  class logits (background=0, shadow=1)
        min_pixels:  Minimum connected-component area to keep (pixels)

    Returns:
        filtered_predictions: Tensor same shape; small shadow components zeroed out.
    """
    filtered = predictions.clone()

    for i in range(predictions.shape[0]):
        # ── Argmax-based binary mask (correct) ────────────────────────────────
        # Works regardless of whether logits are large (vanilla CE) or small
        # (ISW-regularized). Only the relative ordering of the two class logits
        # matters for classification.
        pred_binary = (torch.argmax(predictions[i], dim=0) == 1).cpu().numpy().astype(np.uint8)

        # ── Remove connected components smaller than min_pixels ───────────────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pred_binary, connectivity=8
        )

        for label in range(1, num_labels):          # label 0 is background
            if stats[label, cv2.CC_STAT_AREA] < min_pixels:
                pred_binary[labels == label] = 0

        # ── Write filtered mask back into both logit channels ─────────────────
        # Set shadow channel to +1 and background channel to -1 where shadow
        # survives, and the reverse where it was removed. This preserves the
        # downstream argmax contract without introducing arbitrary large values.
        shadow_mask    = torch.from_numpy(pred_binary).bool().to(predictions.device)
        background_mask = ~shadow_mask

        filtered[i, 1] =  shadow_mask.float()      # shadow logit: 1 where predicted shadow
        filtered[i, 0] =  background_mask.float()  # background logit: 1 where NOT shadow

    return filtered


def filter_small_masks(masks, min_pixels=10):
    """
    Remove small connected components from already-binarized masks.

    Args:
        masks:      Tensor [B, H, W] or [B, 1, H, W]  binary values {0, 1}
        min_pixels: Minimum component area to keep

    Returns:
        filtered_masks: Tensor same shape
    """
    if masks.dim() == 3:
        masks          = masks.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    filtered = masks.clone()

    for i in range(masks.shape[0]):
        mask_binary = masks[i, 0].cpu().numpy().astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_binary, connectivity=8
        )

        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_pixels:
                mask_binary[labels == label] = 0

        filtered[i, 0] = torch.from_numpy(mask_binary).to(masks.device)

    if squeeze_output:
        filtered = filtered.squeeze(1)

    return filtered