"""
shadowtransfer.metrics — the single metric implementation for new work.

CONTRACT.md rule zero. Every new script imports from here. The 30+ legacy
implementations elsewhere in the tree are frozen: they produced the published
numbers and are the audit trail. Nothing here refactors, renames or fixes them.

Assembly rule: every function below is either LIFTED verbatim from a named
source (function body copied, behaviour bit-for-bit, only imports changed) or
marked NEW. Anything unmarked is a defect.

=============================================================================
PROVENANCE — one line per public name
=============================================================================

Boundary band, don't-care confusion counts, size/contrast strata
  DetailedEvaluator ............ LIFTED-FROM python/mamnet/utils/evaluation_detailed.py
                                 078bcc7eb338a1eff4570f1f05ee8740
                                 (byte-identical in mamnet, oglanet, dinov3)

Strict confusion counts and strict IoU / mIoU
  ShadowMetrics ................ LIFTED-FROM python/mamnet/utils/metrics.py
                                 04d778df4d62c8685bdc50edcdc018e7
                                 (byte-identical in all three arch trees)
  evaluate_model ............... LIFTED-FROM python/mamnet/utils/metrics.py
                                 04d778df4d62c8685bdc50edcdc018e7

<10 px connected-component filter applied to the published predictions
  filter_small_predictions ..... LIFTED-FROM python/mamnet/utils/postprocessing.py
                                 015ccd5a97c3f1abdeeb65ffaa76a7ba
                                 (byte-identical in all three arch trees)
  filter_small_masks ........... LIFTED-FROM python/mamnet/utils/postprocessing.py
                                 015ccd5a97c3f1abdeeb65ffaa76a7ba

Risk-coverage, 0/1 AURC
  COVERAGE_GRID ................ LIFTED-FROM python/final_loco/composition_analysis/
                                 s1_risk_coverage_select_predict_gap.py
                                 32d53b1fd4338a9d8e8eea3f298e1ac1
  per_image_rc ................. LIFTED-FROM s1_risk_coverage_select_predict_gap.py
                                 32d53b1fd4338a9d8e8eea3f298e1ac1
  aurc_trapz19 ................. LIFTED-FROM s1_risk_coverage_select_predict_gap.py
                                 32d53b1fd4338a9d8e8eea3f298e1ac1
                                 (source name `aurc`; renamed per the P0a brief
                                 so both live AURC definitions coexist)
  aurc_mean20 .................. LIFTED-FROM python/mamnet/train_mamnet_sib.py
                                 ecf894293f4f1b3c2263adbd1b77af62
                                 (source name `compute_aurc`; renamed for the
                                 same reason. Twins in oglanet/dinov3 SIB
                                 trainers are the same function.)

  The two AURC definitions are NOT on the same scale and both are live in the
  tree. CONTRACT.md §3 carries a blank for which one is canonical. This module
  ships both, side by side, and picks neither.

Cross-entropy risk-coverage (App G.5)
  EPS_CE ....................... LIFTED-FROM python/final_loco/composition_analysis/
                                 f5_cross_entropy_aurc.py
                                 cb9d7f84a59eb34a08c4f6bc6999407f (source name `EPS`)
  per_image_rc_ce .............. LIFTED-FROM f5_cross_entropy_aurc.py
                                 cb9d7f84a59eb34a08c4f6bc6999407f
  aurc_ce_trapz19 .............. LIFTED-FROM f5_cross_entropy_aurc.py
                                 cb9d7f84a59eb34a08c4f6bc6999407f
                                 (source name `aurc`; renamed. Body is identical
                                 to s1's `aurc`; kept separate so a future edit
                                 to one source cannot silently move the other.)

Logits and per-population logit means
  P_CLIP ....................... LIFTED-FROM python/final_loco/composition_analysis/
                                 s2_f1_logit_mean_decomp.py
                                 4bba25cc665e30aabdf1e1ad02b19887
  probs_to_logits .............. LIFTED-FROM s2_f1_logit_mean_decomp.py
                                 4bba25cc665e30aabdf1e1ad02b19887
                                 (byte-identical copy also in s2_f5_pop_stratified_ece.py
                                 cef727072596abb601a33c0dcab3d058)
  population_means ............. LIFTED-FROM s2_f1_logit_mean_decomp.py
                                 4bba25cc665e30aabdf1e1ad02b19887

Class-stratified ECE (§4.3 / App G.4)
  N_BINS_HALF .................. LIFTED-FROM python/final_loco/composition_analysis/
                                 s2_f5_pop_stratified_ece.py
                                 cef727072596abb601a33c0dcab3d058
  stratified_ece ............... LIFTED-FROM s2_f5_pop_stratified_ece.py
                                 cef727072596abb601a33c0dcab3d058
  apply_T ...................... LIFTED-FROM s2_f5_pop_stratified_ece.py
                                 cef727072596abb601a33c0dcab3d058
  apply_class_T ................ LIFTED-FROM s2_f5_pop_stratified_ece.py
                                 cef727072596abb601a33c0dcab3d058
  compute_ece .................. LIFTED-FROM python/mamnet/train_mamnet_sib.py
                                 ecf894293f4f1b3c2263adbd1b77af62
                                 (the SIB-trainer 15-bin ECE; a second live
                                 definition, kept for the same reason as AURC)

Mean predicted probability by GT class (Fig 2's p̄_S)
  pbar_populations ............. LIFTED-FROM python/select_figure2_tiles.py
                                 920fbe83688d69535eb717fc20f34de4
                                 (lines 193-198; the only implementation of
                                 Fig 2's p̄_S / p̄_B in the tree)

Recovery ratio R
  compute_per_image_recovery ... LIFTED-FROM python/final_loco/evaluate_recovery_robust.py
                                 4dc2f8741161ff338a2fef6a5cb217a6

-----------------------------------------------------------------------------
NEW — no implementation of these exists anywhere on disk as of 2026-07-27
-----------------------------------------------------------------------------
  boundary_valid_mask .......... NEW. Thin wrapper: instantiates the LIFTED
                                 DetailedEvaluator at band width K and calls its
                                 _compute_valid_mask. Adds no arithmetic.
  per_image_counts ............. NEW. Thin wrapper: drives the LIFTED
                                 DetailedEvaluator over one image and returns its
                                 own per-image accumulators. Adds no arithmetic.
  logit_mean_tp ................ NEW. probs_to_logits (LIFTED, verbatim) masked
  logit_mean_fp ................ NEW.   by each of the four confusion cells on
  logit_mean_fn ................ NEW.   the valid region, then a plain mean.
  logit_mean_tn ................ NEW.   No confusion-cell implementation exists.
  logit_means_confusion ........ NEW. All four in one pass.
  n_gt_fg_valid ................ NEW. Plain count on the valid region.
  empty_fg_valid ............... NEW. Plain count on the valid region.
  n_pred_pos ................... NEW. Plain count on the valid region.
  threshold_and_filter ......... NEW. Composes a 0.5 threshold with the LIFTED
                                 filter_small_predictions; adds no arithmetic.

-----------------------------------------------------------------------------
FLOAT16 SATURATION — what probs_to_logits does at the endpoints
-----------------------------------------------------------------------------
The stored probability maps in data/Test_img_probs/ are float16. float16 cannot
represent a probability closer to 1.0 than 0.99951171875 (the next value below
1.0), nor closer to 0.0 than ~6e-8; anything nearer rounds to exactly 1.0 or
exactly 0.0 on write. Those saturated values are common in the published cells.

probs_to_logits clips to P_CLIP = (1e-6, 1 - 1e-6) BEFORE the log-odds, so:

    p = 0.0  ->  log(1e-6 / (1 - 1e-6))          = -13.815510557964274
    p = 1.0  ->  log((1 - 1e-6) / (1 - (1-1e-6)))= +13.815510557964272

i.e. saturated pixels contribute a finite +-13.8155 nats, not +-inf. The two
endpoints are not exactly symmetric in the last ulp because (1 - 1e-6) is not
exactly representable in float64. This is the existing clipping behaviour of
s2_f1_logit_mean_decomp.py; it is reproduced verbatim and no epsilon is invented
here. Every logit_mean_* value is therefore bounded by +-13.8155, and a
population that is entirely saturated returns exactly that bound.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch
import cv2
from scipy.ndimage import distance_transform_edt

# The LIFTED DetailedEvaluator body calls compute_shadow_contrast. That helper is
# byte-identical across the three arch trees (8983273da4a390d711b633d2db97aaf4);
# import it rather than copy it.
_MAMNET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mamnet")
if _MAMNET_DIR not in sys.path:
    sys.path.insert(0, _MAMNET_DIR)
from utils.contrast_utils import compute_shadow_contrast  # noqa: E402


# =============================================================================
# LIFTED-FROM python/mamnet/utils/evaluation_detailed.py
#             078bcc7eb338a1eff4570f1f05ee8740
# =============================================================================

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


# =============================================================================
# LIFTED-FROM python/mamnet/utils/metrics.py
#             04d778df4d62c8685bdc50edcdc018e7
# =============================================================================

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


# =============================================================================
# LIFTED-FROM python/mamnet/utils/postprocessing.py
#             015ccd5a97c3f1abdeeb65ffaa76a7ba
# =============================================================================

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


# =============================================================================
# LIFTED-FROM python/final_loco/composition_analysis/
#             s1_risk_coverage_select_predict_gap.py
#             32d53b1fd4338a9d8e8eea3f298e1ac1
# =============================================================================

COVERAGE_GRID = np.linspace(0.10, 1.00, 19)


def per_image_rc(gt, prob, mode):
    gt_flat = gt.ravel().astype(bool); p_flat = prob.ravel()
    if mode == 'all_pixels':
        pred = (p_flat > 0.5)
        err  = (pred != gt_flat).astype(np.float32)
        conf = np.maximum(p_flat, 1.0 - p_flat)
    elif mode == 'gt_shadow':
        if gt_flat.sum() < 50: return None
        p_flat = p_flat[gt_flat]; err = (p_flat <= 0.5).astype(np.float32); conf = p_flat
    elif mode == 'gt_bg':
        if (~gt_flat).sum() < 50: return None
        p_flat = p_flat[~gt_flat]; err = (p_flat > 0.5).astype(np.float32); conf = 1.0 - p_flat
    else: raise ValueError(mode)
    n = len(conf)
    if n < 50: return None
    order = np.argsort(-conf); cum_err = np.cumsum(err[order])
    rc = np.empty(len(COVERAGE_GRID), dtype=np.float32)
    for i, c in enumerate(COVERAGE_GRID):
        k = max(int(np.ceil(c * n)), 1)
        rc[i] = cum_err[k - 1] / k
    return rc


def aurc_trapz19(rc):
    """Source name: `aurc`. Renamed only; body verbatim."""
    if rc is None or np.any(np.isnan(rc)): return float('nan')
    return float(np.trapezoid(rc, COVERAGE_GRID))


# =============================================================================
# LIFTED-FROM python/mamnet/train_mamnet_sib.py
#             ecf894293f4f1b3c2263adbd1b77af62
# =============================================================================

def aurc_mean20(confidences, correct, n_coverage=20):
    """
    Source name: `compute_aurc`. Renamed only; body verbatim.

    Area Under Risk-Coverage curve.
    Sort by confidence (desc), error rate at coverage c uses top c-fraction.
    """
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    if len(confidences) == 0:
        return float('nan')
    sort_idx = np.argsort(-confidences)
    sorted_correct = correct[sort_idx]
    coverage_grid = np.linspace(0.10, 1.0, n_coverage)
    aurc = 0.0
    for c in coverage_grid:
        k = max(1, int(c * len(confidences)))
        aurc += 1.0 - sorted_correct[:k].mean()
    return aurc / n_coverage


def compute_ece(confidences, correct, n_bins=15):
    """Expected Calibration Error."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(float)
    if len(confidences) == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(confidences)
    for i in range(n_bins):
        m = (confidences > edges[i]) & (confidences <= edges[i+1])
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(confidences[m].mean() - correct[m].mean())
    return ece


# =============================================================================
# LIFTED-FROM python/final_loco/composition_analysis/f5_cross_entropy_aurc.py
#             cb9d7f84a59eb34a08c4f6bc6999407f
# =============================================================================

EPS_CE = 1e-7          # source name: EPS


def per_image_rc_ce(gt, prob, mode):
    gt_flat = gt.ravel().astype(bool); p_flat = np.clip(prob.ravel().astype(np.float32), EPS_CE, 1-EPS_CE)
    if mode == 'gt_shadow':
        if gt_flat.sum() < 50: return None
        p = p_flat[gt_flat]; loss = -np.log(p); conf = p
    elif mode == 'gt_bg':
        if (~gt_flat).sum() < 50: return None
        p = p_flat[~gt_flat]; loss = -np.log(1-p); conf = 1-p
    elif mode == 'all_pixels':
        target = gt_flat.astype(np.float32)
        loss = -(target * np.log(p_flat) + (1-target) * np.log(1-p_flat))
        conf = np.maximum(p_flat, 1-p_flat)
    else: raise ValueError(mode)
    n = len(conf)
    if n < 50: return None
    order = np.argsort(-conf); cum = np.cumsum(loss[order])
    rc = np.empty(len(COVERAGE_GRID), dtype=np.float32)
    for i, c in enumerate(COVERAGE_GRID):
        k = max(int(np.ceil(c * n)), 1); rc[i] = cum[k-1] / k
    return rc


def aurc_ce_trapz19(rc):
    """Source name: `aurc` (f5). Renamed only; body verbatim."""
    return float(np.trapezoid(rc, COVERAGE_GRID))


# =============================================================================
# LIFTED-FROM python/final_loco/composition_analysis/s2_f1_logit_mean_decomp.py
#             4bba25cc665e30aabdf1e1ad02b19887
# =============================================================================

P_CLIP = (1e-6, 1.0 - 1e-6)


def probs_to_logits(p):
    p = np.clip(p, *P_CLIP)
    return np.log(p / (1.0 - p))


def population_means(prob, gt):
    """Mean logit on each of TP_pred, FP_pred, FN_pred, TN_pred."""
    z = probs_to_logits(prob)
    pred_pos = z > 0
    gt_pos = gt > 0
    out = {}
    masks = {
        'TP_pred': pred_pos & gt_pos,
        'FP_pred': pred_pos & ~gt_pos,
        'FN_pred': ~pred_pos & gt_pos,
        'TN_pred': ~pred_pos & ~gt_pos,
    }
    for name, m in masks.items():
        if m.sum() > 0:
            out[f'{name}_mean'] = float(z[m].mean())
            out[f'{name}_count'] = int(m.sum())
        else:
            out[f'{name}_mean'] = float('nan')
            out[f'{name}_count'] = 0
    return out


# =============================================================================
# LIFTED-FROM python/final_loco/composition_analysis/s2_f5_pop_stratified_ece.py
#             cef727072596abb601a33c0dcab3d058
# =============================================================================

N_BINS_HALF = 10           # bins per half (10 over [0,0.5], 10 over [0.5,1])


def apply_T(prob, T):
    z = probs_to_logits(prob)
    return 1.0 / (1.0 + np.exp(-z / T))


def apply_class_T(prob, Tp, Tn):
    z = probs_to_logits(prob)
    return 1.0 / (1.0 + np.exp(-np.where(z > 0, z / Tp, z / Tn)))


def stratified_ece(eval_data, transform=None):
    """Returns ECE_pred_pos, ECE_pred_neg, n_pos_pixels, n_neg_pixels."""
    edges_neg = np.linspace(0, 0.5, N_BINS_HALF + 1)
    edges_pos = np.linspace(0.5, 1.0, N_BINS_HALF + 1)

    sums = {'pos': defaultdict(lambda: [0.0, 0.0, 0]),
            'neg': defaultdict(lambda: [0.0, 0.0, 0])}

    for stem, gt, prob in eval_data:
        p = transform(prob) if transform else prob
        z_sign = (p > 0.5).astype(np.uint8)
        y = gt.ravel().astype(np.float32)
        p_flat = p.ravel()

        # pred-positive pixels
        m_pos = z_sign.ravel() == 1
        if m_pos.any():
            pp = p_flat[m_pos]; yy = y[m_pos]
            idx = np.clip(np.digitize(pp, edges_pos) - 1, 0, N_BINS_HALF - 1)
            for b in range(N_BINS_HALF):
                mask = idx == b
                n = int(mask.sum())
                if n > 0:
                    sums['pos'][b][0] += float(pp[mask].sum())
                    sums['pos'][b][1] += float(yy[mask].sum())
                    sums['pos'][b][2] += n

        # pred-negative pixels
        m_neg = z_sign.ravel() == 0
        if m_neg.any():
            pp = p_flat[m_neg]; yy = y[m_neg]
            idx = np.clip(np.digitize(pp, edges_neg) - 1, 0, N_BINS_HALF - 1)
            for b in range(N_BINS_HALF):
                mask = idx == b
                n = int(mask.sum())
                if n > 0:
                    sums['neg'][b][0] += float(pp[mask].sum())
                    sums['neg'][b][1] += float(yy[mask].sum())
                    sums['neg'][b][2] += n

    def collapse(side):
        N = sum(s[2] for s in sums[side].values())
        if N == 0: return float('nan'), 0
        ece = 0.0
        for b, (psum, ysum, n) in sums[side].items():
            avg_p = psum / n
            avg_y = ysum / n
            ece += (n / N) * abs(avg_p - avg_y)
        return float(ece), N

    ece_pos, n_pos = collapse('pos')
    ece_neg, n_neg = collapse('neg')
    return ece_pos, ece_neg, n_pos, n_neg


# =============================================================================
# LIFTED-FROM python/select_figure2_tiles.py
#             920fbe83688d69535eb717fc20f34de4   (lines 193-198)
# =============================================================================

def pbar_populations(prob, gt):
    """
    Fig 2's p̄_S and p̄_B: mean predicted shadow probability inside GT shadow
    and inside GT background. Source expression:
        m["p_loco_S"] = float(loco[shadow].mean())
        m["p_loco_B"] = float(loco[bg].mean())
    """
    shadow = gt > 0
    bg = ~shadow
    return (float(prob[shadow].mean()) if shadow.any() else float('nan'),
            float(prob[bg].mean()) if bg.any() else float('nan'))


# =============================================================================
# LIFTED-FROM python/final_loco/evaluate_recovery_robust.py
#             4dc2f8741161ff338a2fef6a5cb217a6
# =============================================================================

def compute_per_image_recovery(exp_ious, loco_ious, upper_ious,
                                min_gap=0.02):
    """
    Compute per-image recovery ratio.

    R_i = (exp_IoU_i - loco_IoU_i) / (upper_IoU_i - loco_IoU_i)

    Args:
        exp_ious:   list of per-image IoU for experiment condition
        loco_ious:  list of per-image IoU for LOCO baseline
        upper_ious: list of per-image IoU for upper bound
        min_gap:    minimum |upper - loco| to include image (avoids
                    division by near-zero). Default 0.02.

    Returns:
        R_values:     np.array of valid R values
        valid_indices: list of indices into original arrays
        excluded_info: dict with exclusion counts and reasons
    """
    n = len(exp_ious)
    assert len(loco_ious) == n and len(upper_ious) == n, \
        f"Length mismatch: exp={n}, loco={len(loco_ious)}, upper={len(upper_ious)}"

    R_values = []
    valid_indices = []
    n_nan = 0
    n_small_gap = 0

    for i in range(n):
        e, l, u = exp_ious[i], loco_ious[i], upper_ious[i]

        if any(np.isnan(v) for v in [e, l, u]):
            n_nan += 1
            continue

        gap = u - l
        if abs(gap) < min_gap:
            n_small_gap += 1
            continue

        R_i = (e - l) / gap
        R_values.append(R_i)
        valid_indices.append(i)

    excluded_info = {
        'total_images': n,
        'valid_images': len(R_values),
        'excluded_nan': n_nan,
        'excluded_small_gap': n_small_gap,
        'min_gap_threshold': min_gap,
    }

    return np.array(R_values), valid_indices, excluded_info


# =============================================================================
# NEW — no implementation exists anywhere in the tree as of 2026-07-27
# =============================================================================

def threshold_and_filter(prob, min_pixels=10):
    """
    NEW. Composition, no new arithmetic.

    Threshold a single-channel float probability map at 0.5 and apply the
    LIFTED filter_small_predictions.

    The two-channel stack fed to filter_small_predictions is [1 - p, p], so its
    torch.argmax(dim=0) == 1 exactly when p > 0.5 (torch.argmax returns the
    FIRST maximal index, so p == 0.5 resolves to background). That reproduces
    the argmax call that run_inference.py made on the real class logits.

    Args:
        prob: float array [H, W], P(shadow)

    Returns:
        pred: uint8 array [H, W] in {0, 1}
    """
    p = np.asarray(prob, dtype=np.float32)
    stack = torch.from_numpy(np.stack([1.0 - p, p], axis=0)).unsqueeze(0)
    filtered = filter_small_predictions(stack, min_pixels=min_pixels)
    return (torch.argmax(filtered[0], dim=0) == 1).cpu().numpy().astype(np.uint8)


def boundary_valid_mask(target, boundary_tolerance):
    """
    NEW. Thin wrapper, no new arithmetic.

    Returns the LIFTED DetailedEvaluator's don't-care valid mask at band width
    K = boundary_tolerance. K = 0 gives a 1x1 kernel, so erode and dilate are
    both the identity, the band is empty, and every pixel is valid — the strict
    population.
    """
    return DetailedEvaluator(
        boundary_tolerance=boundary_tolerance)._compute_valid_mask(target)


def per_image_counts(pred, target, boundary_tolerance=2):
    """
    NEW. Thin wrapper, no new arithmetic.

    Drives the LIFTED DetailedEvaluator over one (pred, target) pair and returns
    its own per-image accumulators:

        strict  -> DetailedEvaluator._per_image_strict[-1]   (K = 0, all pixels)
        tol     -> DetailedEvaluator._per_image_dc[-1]       (band width K)

    Both tuples are ordered (tp, fp, tn, fn), matching the source.

    Returns:
        dict with tp_str/fp_str/tn_str/fn_str, tp_tol/fp_tol/tn_tol/fn_tol,
        n_valid_px, and the valid mask itself.
    """
    ev = DetailedEvaluator(boundary_tolerance=boundary_tolerance)
    ev.update(torch.from_numpy(np.asarray(pred)[None].astype(np.int64)),
              torch.from_numpy(np.asarray(target)[None].astype(np.int64)))
    tp_s, fp_s, tn_s, fn_s = ev._per_image_strict[-1]
    tp_t, fp_t, tn_t, fn_t = ev._per_image_dc[-1]
    return {
        'tp_str': tp_s, 'fp_str': fp_s, 'tn_str': tn_s, 'fn_str': fn_s,
        'tp_tol': tp_t, 'fp_tol': fp_t, 'tn_tol': tn_t, 'fn_tol': fn_t,
        'n_valid_px': tp_t + fp_t + tn_t + fn_t,
        'valid_mask': ev._compute_valid_mask(np.asarray(target)),
    }


def logit_means_confusion(prob, target, pred, valid_mask):
    """
    NEW. probs_to_logits (LIFTED, verbatim) masked by each of the four confusion
    cells on the valid region, then a plain mean over each cell.

    The confusion cells are built from the FINAL prediction (0.5 threshold plus
    the <10 px filter), not from the sign of the logit. That differs from
    s2_f1_logit_mean_decomp.population_means, which partitions on z > 0 over the
    whole image with no filter and no valid region; population_means is lifted
    above and left alone.

    Saturated float16 inputs give exactly -13.815510557964274 (p = 0) and
    +13.815510557964272 (p = 1); see the module docstring.

    Returns:
        dict with logit_mean_tp / _fp / _fn / _tn. A cell with no pixels on the
        valid region returns nan.
    """
    z = probs_to_logits(np.asarray(prob, dtype=np.float64))
    pred_pos = np.asarray(pred) > 0
    gt_pos = np.asarray(target) > 0
    v = np.asarray(valid_mask, dtype=bool)
    cells = {
        'logit_mean_tp': pred_pos & gt_pos & v,
        'logit_mean_fp': pred_pos & ~gt_pos & v,
        'logit_mean_fn': ~pred_pos & gt_pos & v,
        'logit_mean_tn': ~pred_pos & ~gt_pos & v,
    }
    return {name: (float(z[m].mean()) if m.any() else float('nan'))
            for name, m in cells.items()}


def logit_mean_tp(prob, target, pred, valid_mask):
    """NEW. See logit_means_confusion."""
    return logit_means_confusion(prob, target, pred, valid_mask)['logit_mean_tp']


def logit_mean_fp(prob, target, pred, valid_mask):
    """NEW. See logit_means_confusion."""
    return logit_means_confusion(prob, target, pred, valid_mask)['logit_mean_fp']


def logit_mean_fn(prob, target, pred, valid_mask):
    """NEW. See logit_means_confusion."""
    return logit_means_confusion(prob, target, pred, valid_mask)['logit_mean_fn']


def logit_mean_tn(prob, target, pred, valid_mask):
    """NEW. See logit_means_confusion."""
    return logit_means_confusion(prob, target, pred, valid_mask)['logit_mean_tn']


def n_gt_fg_valid(target, valid_mask):
    """NEW. Ground-truth shadow pixels on the valid region."""
    return int((np.asarray(target) > 0)[np.asarray(valid_mask, dtype=bool)].sum())


def empty_fg_valid(target, valid_mask):
    """NEW. True when n_gt_fg_valid == 0."""
    return bool(n_gt_fg_valid(target, valid_mask) == 0)


def n_pred_pos(pred, valid_mask):
    """NEW. Predicted-positive pixel count on the valid region."""
    return int((np.asarray(pred) > 0)[np.asarray(valid_mask, dtype=bool)].sum())


# =============================================================================
# LIFTED-FROM python/final_loco/evaluate_recovery_robust.py
#             4dc2f8741161ff338a2fef6a5cb217a6   (bootstrap_ci, lines 44-74)
#
# Added for arm E Part 2 (Experiment A): "Report median R per city with an
# image-cluster bootstrap 95% CI, B = 10,000." Each valid image already
# contributes exactly one R value (compute_per_image_recovery above), so an
# "image-cluster" bootstrap over those R values IS standard image-level
# resampling with replacement -- exactly what this function does. This is a
# NEW lift into the canonical module, not an extension of the legacy
# paired_bootstrap_gap/bh_fdr path CLAUDE.md says not to touch (that is a
# different statistic, the pooled mIoU gap, not per-image recovery R).
# =============================================================================

def bootstrap_ci(values, stat_fn=np.median, n_boot=10000,
                 ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: 1D array of observations
        stat_fn: function to compute (default: np.median)
        n_boot: number of bootstrap iterations
        ci: confidence level (default: 0.95)
        seed: random seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.RandomState(seed)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_stats[b] = stat_fn(values[idx])

    boot_stats.sort()
    alpha = (1 - ci) / 2
    lo = boot_stats[int(alpha * n_boot)]
    hi = boot_stats[int((1 - alpha) * n_boot) - 1]
    return (float(stat_fn(values)), float(lo), float(hi))
