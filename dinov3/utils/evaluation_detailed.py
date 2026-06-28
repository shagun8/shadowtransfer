"""
Detailed evaluation module for shadow detection.
Implements:
- Size-stratified metrics (miss rate by shadow size)
- Contrast-stratified metrics (by brightness contrast)
- Boundary-tolerant metrics (±Kpx don't-care zone)
- FP/FN spatial analysis
"""

import numpy as np
import torch
import cv2
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contrast_utils import compute_shadow_contrast


class DetailedEvaluator:
    """
    Comprehensive evaluator for shadow detection with detailed analysis.
    
    Boundary-tolerant evaluation uses a ±K pixel don't-care zone around GT
    shadow boundaries. Pixels in this band are excluded from evaluation entirely,
    since GT labeling is inherently imprecise at boundaries.
    """
    
    def __init__(self, boundary_tolerance=2):
        """
        Args:
            boundary_tolerance: Pixels within this distance of GT boundary are
                                excluded from evaluation (default: 5px)
        """
        self.boundary_tolerance = boundary_tolerance
        # Pre-create morphological kernel (reused every call)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (boundary_tolerance * 2 + 1, boundary_tolerance * 2 + 1)
        )
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        # ===== Overall STRICT metrics =====
        self.tp_total = 0
        self.fp_total = 0
        self.tn_total = 0
        self.fn_total = 0
        self._per_image_strict   = []   # list of per-image (tp,fp,tn,fn) tuples
        self._per_image_dc       = []   # list of per-image (tp,fp,tn,fn) tuples (don't-care)
        
        # ===== Overall DON'T-CARE ZONE metrics =====
        # Pixels in the ±K band around GT boundaries are excluded entirely
        self.tp_dc = 0
        self.fp_dc = 0
        self.tn_dc = 0
        self.fn_dc = 0
        self.num_excluded = 0  # Track how many pixels are in the band
        
        # ===== Size-stratified (strict) =====
        self.size_bins = {
            'tiny': [],      # <20px
            'small': [],     # 20-50px
            'medium': [],    # 50-200px
            'large': []      # >200px
        }
        
        # ===== Size-stratified (don't-care zone) =====
        self.size_bins_dc = {
            'tiny': [],
            'small': [],
            'medium': [],
            'large': []
        }
        
        # ===== Contrast-stratified (strict) =====
        self.contrast_bins = {
            'low': [],       # <0.2
            'medium': [],    # 0.2-0.4
            'high': []       # >0.4
        }

        # ===== Contrast-stratified (don't-care zone) =====
        self.contrast_bins_dc = {
            'low': [],
            'medium': [],
            'high': []
        }

        # ===== Boundary FP/FN spatial analysis =====
        self.fp_distances = []
        self.fn_distances = []
    
    def _compute_valid_mask(self, target):
        """
        Compute the valid evaluation mask by excluding the ±K px band
        around GT shadow boundaries.
        
        Uses fast cv2 morphological ops (erode + dilate) instead of
        distance_transform_edt.
        
        Args:
            target: Binary GT mask [H, W] with values {0, 1}
            
        Returns:
            valid_mask: Boolean [H, W], True = evaluate this pixel,
                        False = don't-care (in boundary band)
        """
        target_uint8 = target.astype(np.uint8)
        
        # Inner region: erode GT → pixels definitely inside shadow
        eroded = cv2.erode(target_uint8, self._kernel)
        # Outer region: dilate GT → pixels definitely outside shadow (beyond this)
        dilated = cv2.dilate(target_uint8, self._kernel)
        
        # Don't-care band = dilated - eroded (the ambiguous zone)
        band = (dilated - eroded) > 0
        
        # Valid = everything NOT in the band
        return ~band
    
    def update(self, predictions, targets, images=None):
        """
        Update evaluation metrics.
        
        Args:
            predictions: Binary predictions [B, H, W] with values {0, 1}
            targets: Ground truth [B, H, W] with values {0, 1}
            images: Original images [B, C, H, W] for contrast computation (optional)
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        if images is not None:
            images = images.cpu().numpy()
        
        B = predictions.shape[0]
        
        for b in range(B):
            pred = predictions[b]
            target = targets[b]
            image = images[b] if images is not None else None
            
            # ---- Overall STRICT confusion matrix ----
            tp = np.logical_and(pred == 1, target == 1).sum()
            fp = np.logical_and(pred == 1, target == 0).sum()
            tn = np.logical_and(pred == 0, target == 0).sum()
            fn = np.logical_and(pred == 0, target == 1).sum()
            
            self.tp_total += tp
            self.fp_total += fp
            self.tn_total += tn
            self.fn_total += fn
            self._per_image_strict.append((int(tp), int(fp), int(tn), int(fn)))
            
            # ---- Compute don't-care valid mask ONCE per image ----
            valid_mask = self._compute_valid_mask(target)
            
            # ---- Overall DON'T-CARE ZONE confusion matrix ----
            self._update_dontcare_overall(pred, target, valid_mask)
            
            # ---- Size-stratified analysis (strict + don't-care) ----
            self._update_size_stratified(pred, target, image, valid_mask)
            
            # ---- Boundary FP/FN spatial analysis ----
            self._update_boundary_analysis(pred, target)
    
    def _update_dontcare_overall(self, pred, target, valid_mask):
        """Update overall metrics excluding don't-care band pixels"""
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        self.tp_dc += np.logical_and(pred_valid == 1, target_valid == 1).sum()
        self.fp_dc += np.logical_and(pred_valid == 1, target_valid == 0).sum()
        self.tn_dc += np.logical_and(pred_valid == 0, target_valid == 0).sum()
        self.fn_dc += np.logical_and(pred_valid == 0, target_valid == 1).sum()
        self.num_excluded += (~valid_mask).sum()
        self._per_image_dc.append((
            int(np.logical_and(pred_valid == 1, target_valid == 1).sum()),
            int(np.logical_and(pred_valid == 1, target_valid == 0).sum()),
            int(np.logical_and(pred_valid == 0, target_valid == 0).sum()),
            int(np.logical_and(pred_valid == 0, target_valid == 1).sum()),
        ))
    
    def _update_size_stratified(self, pred, target, image, valid_mask):
        """Update size-stratified metrics for both strict and don't-care"""
        # Find connected components in ground truth
        target_uint8 = target.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(target_uint8)
        
        for label_id in range(1, num_labels):
            shadow_mask = (labels == label_id)
            size = shadow_mask.sum()
            
            # Categorize by size
            if size < 20:
                category = 'tiny'
            elif size < 50:
                category = 'small'
            elif size < 200:
                category = 'medium'
            else:
                category = 'large'
            
            # ---- STRICT metrics ----
            shadow_pred_strict = pred[shadow_mask]
            detected_strict = shadow_pred_strict.sum() > 0
            iou_strict = self._compute_iou_single(pred, shadow_mask)
            
            # ---- DON'T-CARE metrics ----
            # Only evaluate on valid pixels within this shadow region
            shadow_valid = np.logical_and(shadow_mask, valid_mask)
            valid_px_count = shadow_valid.sum()
            
            if valid_px_count > 0:
                detected_dc = pred[shadow_valid].sum() > 0
                iou_dc = self._compute_iou_single_masked(pred, shadow_mask, valid_mask)
            else:
                # Entire shadow is in the don't-care band (very tiny shadow)
                detected_dc = True  # Don't penalize
                iou_dc = 1.0
            
            # Compute contrast if image available
            contrast = None
            if image is not None:
                img_hwc = np.transpose(image, (1, 2, 0))
                
                # Handle 4-channel images (Task 2: RGB + Contrast)
                if img_hwc.shape[2] == 4:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_hwc[:, :, :3] = img_hwc[:, :, :3] * std + mean
                    img_hwc = img_hwc[:, :, :3]
                else:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_hwc = img_hwc * std + mean
                
                img_hwc = np.clip(img_hwc, 0, 1)
                contrast = compute_shadow_contrast(shadow_mask.astype(np.float32), img_hwc)
            
            # Store STRICT
            self.size_bins[category].append({
                'size': size,
                'detected': detected_strict,
                'iou': iou_strict,
                'contrast': contrast
            })
            
            # Store DON'T-CARE
            self.size_bins_dc[category].append({
                'size': size,
                'detected': detected_dc,
                'iou': iou_dc,
                'contrast': contrast
            })
            
            # Contrast-stratified (if available)
            if contrast is not None:
                if contrast < 0.2:
                    contrast_cat = 'low'
                elif contrast < 0.4:
                    contrast_cat = 'medium'
                else:
                    contrast_cat = 'high'
                
                self.contrast_bins[contrast_cat].append({
                    'size': size,
                    'detected': detected_strict,
                    'iou': iou_strict,
                    'contrast': contrast
                })
                
                self.contrast_bins_dc[contrast_cat].append({
                    'size': size,
                    'detected': detected_dc,
                    'iou': iou_dc,
                    'contrast': contrast
                })
    
    def _update_boundary_analysis(self, pred, target):
        """Update FP/FN spatial distribution (diagnostic only)"""
        # False positives
        fp_mask = np.logical_and(pred == 1, target == 0)
        if fp_mask.any() and target.any():
            dist_transform = distance_transform_edt(1 - target)
            fp_distances = dist_transform[fp_mask]
            self.fp_distances.extend(fp_distances.tolist())
        
        # False negatives
        fn_mask = np.logical_and(pred == 0, target == 1)
        if fn_mask.any() and target.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(target.astype(np.uint8), kernel)
            eroded = cv2.erode(target.astype(np.uint8), kernel)
            boundary = (dilated - eroded) > 0
            
            dist_transform = distance_transform_edt(1 - boundary)
            fn_distances = dist_transform[fn_mask]
            self.fn_distances.extend(fn_distances.tolist())
    
    def _compute_iou_single(self, pred, target_mask):
        """Compute IoU for a single shadow (strict, all pixels)"""
        pred_mask = pred.astype(bool)
        target_mask = target_mask.astype(bool)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)
    
    def _compute_iou_single_masked(self, pred, target_mask, valid_mask):
        """
        Compute IoU for a single shadow, only on valid (non-band) pixels.
        
        Pixels in the don't-care band are excluded from both intersection
        and union counts.
        """
        pred_bool = pred.astype(bool)
        target_bool = target_mask.astype(bool)
        
        # Restrict to valid pixels only
        pred_valid = np.logical_and(pred_bool, valid_mask)
        target_valid = np.logical_and(target_bool, valid_mask)
        
        intersection = np.logical_and(pred_valid, target_valid).sum()
        union = np.logical_or(pred_valid, target_valid).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)
    
    def compute_metrics(self):
        """Compute all metrics"""
        results = {}
        
        # Overall metrics (strict)
        results['overall'] = self._compute_overall_metrics()
        
        # Size-stratified (strict)
        results['size_stratified'] = self._compute_stratified('size', strict=True)
        
        # Size-stratified (don't-care)
        results['size_stratified_tolerant'] = self._compute_stratified('size', strict=False)
        
        # Contrast-stratified (strict)
        results['contrast_stratified'] = self._compute_stratified('contrast', strict=True)
        
        # Contrast-stratified (don't-care)
        results['contrast_stratified_tolerant'] = self._compute_stratified('contrast', strict=False)
        
        # Boundary-tolerant (strict vs don't-care zone)
        results['boundary_tolerant'] = self._compute_boundary_tolerant_metrics()
        
        # FP/FN spatial analysis
        results['fp_fn_analysis'] = self._compute_fp_fn_analysis()
        
        return results
    
    def _compute_overall_metrics(self):
        def _metrics_from_counts(tp, fp, tn, fn):
            precision = tp / (tp + fp + 1e-10)
            recall    = tp / (tp + fn + 1e-10)
            f1        = 2 * precision * recall / (precision + recall + 1e-10)
            iou       = tp / (tp + fp + fn + 1e-10)
            return precision * 100, recall * 100, f1 * 100, iou * 100

        rows = [_metrics_from_counts(*c) for c in self._per_image_strict]
        if not rows:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
        return {
            'precision': float(np.mean([r[0] for r in rows])),
            'recall':    float(np.mean([r[1] for r in rows])),
            'f1':        float(np.mean([r[2] for r in rows])),
            'iou':       float(np.mean([r[3] for r in rows])),
        }
    
    def _compute_stratified(self, stratify_by, strict=True):
        """
        Compute stratified metrics (works for both size and contrast).
        
        Args:
            stratify_by: 'size' or 'contrast'
            strict: If True use strict bins, else use don't-care bins
        """
        if stratify_by == 'size':
            bins = self.size_bins if strict else self.size_bins_dc
        else:
            bins = self.contrast_bins if strict else self.contrast_bins_dc
        
        results = {}
        for category, shadows in bins.items():
            if len(shadows) == 0:
                continue
            
            total = len(shadows)
            detected = sum(1 for s in shadows if s['detected'])
            miss_rate = (total - detected) / total * 100
            avg_iou = np.mean([s['iou'] for s in shadows]) * 100
            
            entry = {
                'total': total,
                'detected': detected,
                'miss_rate': float(miss_rate),
                'avg_iou': float(avg_iou)
            }
            
            # Add avg contrast for contrast-stratified
            if stratify_by == 'contrast':
                entry['avg_contrast'] = float(np.mean([s['contrast'] for s in shadows]))
            
            results[category] = entry
        
        return results
    
    def _compute_boundary_tolerant_metrics(self):
        def _img_metrics(tp, fp, tn, fn):
            prec = tp / (tp + fp + 1e-10)
            rec  = tp / (tp + fn + 1e-10)
            f1   = 2 * prec * rec / (prec + rec + 1e-10)
            iou  = tp / (tp + fp + fn + 1e-10)
            return prec * 100, rec * 100, f1 * 100, iou * 100

        s_rows  = [_img_metrics(*c) for c in self._per_image_strict]
        dc_rows = [_img_metrics(*c) for c in self._per_image_dc]

        def _avg(rows):
            if not rows:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
            return {
                'precision': float(np.mean([r[0] for r in rows])),
                'recall':    float(np.mean([r[1] for r in rows])),
                'f1':        float(np.mean([r[2] for r in rows])),
                'iou':       float(np.mean([r[3] for r in rows])),
            }

        total_pixels = (self.tp_total + self.fp_total
                        + self.tn_total + self.fn_total)
        tol = _avg(dc_rows)
        tol['pixels_excluded'] = int(self.num_excluded)
        tol['pct_excluded']    = float(
            self.num_excluded / (total_pixels + 1e-10) * 100)

        return {'strict': _avg(s_rows), f'tolerant_{self.boundary_tolerance}px': tol}
    
    def _compute_fp_fn_analysis(self):
        """Compute FP/FN spatial distribution"""
        results = {}
        
        if len(self.fp_distances) > 0:
            fp_distances = np.array(self.fp_distances)
            total_fp = len(fp_distances)
            within_1px = (fp_distances <= 1).sum()
            within_5px = (fp_distances <= 5).sum()
            within_10px = (fp_distances <= 10).sum()
            
            results['fp'] = {
                'total': int(total_fp),
                'within_1px': int(within_1px),
                'within_5px': int(within_5px),
                'within_10px': int(within_10px),
                'pct_within_1px': float(within_1px / total_fp * 100),
                'pct_within_5px': float(within_5px / total_fp * 100),
                'pct_within_10px': float(within_10px / total_fp * 100)
            }
        
        if len(self.fn_distances) > 0:
            fn_distances = np.array(self.fn_distances)
            total_fn = len(fn_distances)
            within_1px = (fn_distances <= 1).sum()
            within_5px = (fn_distances <= 5).sum()
            within_10px = (fn_distances <= 10).sum()
            
            results['fn'] = {
                'total': int(total_fn),
                'within_1px': int(within_1px),
                'within_5px': int(within_5px),
                'within_10px': int(within_10px),
                'pct_within_1px': float(within_1px / total_fn * 100),
                'pct_within_5px': float(within_5px / total_fn * 100),
                'pct_within_10px': float(within_10px / total_fn * 100)
            }
        
        return results


if __name__ == "__main__":
    # Test evaluator
    print("Testing DetailedEvaluator...")
    
    evaluator = DetailedEvaluator(boundary_tolerance=2)
    
    # Create test data
    pred = torch.randint(0, 2, (2, 256, 256))
    target = torch.randint(0, 2, (2, 256, 256))
    images = torch.randn(2, 3, 256, 256)
    
    # Update
    evaluator.update(pred, target, images)
    
    # Compute metrics
    results = evaluator.compute_metrics()
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    print("\nOverall Metrics (Strict):")
    for key, val in results['overall'].items():
        print(f"  {key}: {val:.2f}%")
    
    print("\nSize-Stratified (Strict):")
    for category, metrics in results['size_stratified'].items():
        print(f"  {category}: Miss rate = {metrics['miss_rate']:.1f}%, IoU = {metrics['avg_iou']:.1f}%")
    
    print("\nSize-Stratified (Don't-Care Zone):")
    for category, metrics in results['size_stratified_tolerant'].items():
        print(f"  {category}: Miss rate = {metrics['miss_rate']:.1f}%, IoU = {metrics['avg_iou']:.1f}%")
    
    print("\nBoundary Evaluation:")
    bt = results['boundary_tolerant']
    print(f"  Strict F1:         {bt['strict']['f1']:.2f}%")
    print(f"  Strict IoU:        {bt['strict']['iou']:.2f}%")
    print(f"  Don't-Care F1:     {bt[f'tolerant_{evaluator.boundary_tolerance}px']['f1']:.2f}%")
    print(f"  Don't-Care IoU:    {bt[f'tolerant_{evaluator.boundary_tolerance}px']['iou']:.2f}%")
    print(f"  Pixels excluded:   {bt[f'tolerant_{evaluator.boundary_tolerance}px']['pixels_excluded']} ({bt[f'tolerant_{evaluator.boundary_tolerance}px']['pct_excluded']:.1f}%)")
    
    if 'fp' in results['fp_fn_analysis']:
        print("\nFP Spatial Analysis:")
        fp_info = results['fp_fn_analysis']['fp']
        print(f"  Total FP: {fp_info['total']}")
        print(f"  Within 1px: {fp_info['pct_within_1px']:.1f}%")
        print(f"  Within 5px: {fp_info['pct_within_5px']:.1f}%")