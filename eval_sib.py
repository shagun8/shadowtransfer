"""
eval_sib.py — Unified evaluation for MAMNet / OGLANet / DINOv3 + SIB.

Predictions-only mode (no model loading).  All aggregation is per-image mean.
Primary metric: tolerant ±K px mIOU (default K=2).

Workflow per model type
-----------------------
For each city (chicago, miami, phoenix):
  1. Score Upper Bound predictions → establishes reference stem set
  2. Score all LOCO baseline dirs restricted to those stems
  3. Auto-discover all SIB variant prediction folders; score restricted to
     the same stems
  4. Print per-city comparison tables (baselines first, then SIB variants)
  5. Print recovery ratios for best SIB variant
  6. Print per-image F1 distributions
  7. Bootstrap test: best SIB vs LOCO Vanilla

After all cities:
  8. Rank SIB variants by average tolerant mIOU across cities
  9. Save  eval_{model_type}_{resolution}.json
  10. If eval JSONs for all 3 models exist → print + save cross-model table

SIB folder discovery patterns
------------------------------
  mamnet  : {sib_output_dir}/mamnet_sib_{TAG}__loco_holdout_{city}__{res}
  oglanet : {sib_output_dir}/oglanet_sib_{TAG}__loco_holdout_{city}__{res}
  dinov3  : {sib_output_dir}/dinov3_sib{TAG}_loco_holdout_{city}_{res}_{run}
             where TAG starts with _ (e.g. _haar_vib_aug_ab) and
             run is a numeric suffix (e.g. _1) appended by the training script.

Location: python/eval_sib.py

Usage
-----
  python eval_sib.py \\
      --model_type mamnet \\
      --sib_output_dir /path/to/data/mamnet/outputs/ \\
      --test_img_results_dir /path/to/data/Test_img_results/ \\
      --gt_base_dir /path/to/data/Final_data_test/ \\
      --output_dir /path/to/eval_outputs/ \\
      --resolution highres \\
      --boundary_tolerance 2
"""

import os
import sys
import json
import glob
import re
import argparse
import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_IMG_EXTS    = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
_METRIC_KEYS = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']

# Baseline display labels, in print order
_ORDERED_BASELINE_LABELS = [
    'Upper Bound',
    'LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
    'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA',
]

# Baseline subfolder names inside Test_img_results/loco/{city}/{res}/{model}/
_LOCO_VARIANT_DIRS = {
    'LOCO Vanilla':  'vanilla',
    'LOCO FDA':      'fda',
    'LOCO SegDesic': 'segdesic',
    'LOCO IIM':      'iim',
    'LOCO ISW':      'isw',
    'LOCO MRFP+':    'mrfp_plus',
    'LOCO FADA':     'fada',
}

_MODEL_TYPES = ['mamnet', 'oglanet', 'dinov3']


# ═══════════════════════════════════════════════════════════════════════════════
# Round 2 experiment configuration
#
# Defines pairwise comparison pairs and tag prefixes for the targeted
# Round 2 experiments that aim to find a single winning SIB variant.
# ═══════════════════════════════════════════════════════════════════════════════

_ROUND2_CONFIG = {
    'mamnet': {
        'pairs': [
            # (variant_A, variant_B, question)
            # Δ = A − B;  positive means A is better
            ('M11', 'M1',  'SAG without AB vs Full (SAG+AB)'),
            ('M11', 'M7',  'Add SAG (no AB baseline)'),
            ('M12', 'M7',  'Lower beta (0.0005 vs default)'),
            ('M13', 'M7',  'Passthrough gate value'),
            ('M14', 'M1',  'Module bypass gate vs C4 base (M1)'),
            ('M14', 'M13', 'Module bypass vs VIB-level gate'),
        ],
        'round2_prefixes': ['M11', 'M12', 'M13', 'M14'],
        'context_prefixes': ['M1', 'M7'],
    },
    'oglanet': {
        'pairs': [
            ('O9',  'M7',  'Add SAG (no AB)'),
            ('O9',  'M1',  'Remove AB from M1 (keep SAG)'),
            ('O10', 'M7',  'Lower beta (0.0005 vs default)'),
            ('O11', 'O9',  'Add passthrough gate to SAG'),
            ('O12', 'O9',  'Standard FDA vs attenuated FDA'),
            ('O13', 'M1',  'Module bypass gate vs C4 base (M1)'),
            ('O13', 'O9',  'Module bypass vs best SAG variant'),
        ],
        'round2_prefixes': ['O9', 'O10', 'O11', 'O12', 'O13'],
        'context_prefixes': ['M1', 'M7'],
    },
    'dinov3': {
        'pairs': [
            ('D7',  'D4',  'Low beta (0.002) vs default (0.01)'),
            ('D8',  'D4',  'Aug only (no VIB) vs VIB+Aug'),
            ('D9',  'D4',  'Passthrough gate vs no gate'),
            ('D10', 'D4',  'Mid beta (0.005) vs default (0.01)'),
            ('D11', 'D1',  'Module bypass gate vs D1 base (C4)'),
            ('D11', 'D8',  'Module bypass vs best prior (aug-only)'),
        ],
        'round2_prefixes': ['D7', 'D8', 'D9', 'D10', 'D11'],
        'context_prefixes': ['D1', 'D4'],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-image metric functions
# ═══════════════════════════════════════════════════════════════════════════════

_KERN_CACHE = {}


def _kern(tol):
    if tol not in _KERN_CACHE:
        _KERN_CACHE[tol] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    return _KERN_CACHE[tol]


def _compute_strict(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    prec   = tp / (tp + fp + 1e-10)
    rec    = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * rec / (prec + rec + 1e-10)
    sh_iou = tp / (tp + fp + fn + 1e-10)
    ns_iou = tn / (tn + fp + fn + 1e-10)
    miou   = (sh_iou + ns_iou) / 2
    oa     = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    sh_e   = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0.0
    ns_e   = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0.0
    ber    = (sh_e + ns_e) / 2
    return {
        'OA': float(oa * 100), 'Precision': float(prec * 100),
        'Recall': float(rec * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(sh_iou * 100),
    }


def _compute_tolerant(pred, gt, tol):
    k       = _kern(tol)
    gt_u8   = gt.astype(np.uint8)
    eroded  = cv2.erode(gt_u8, k)
    dilated = cv2.dilate(gt_u8, k)
    valid   = ~((dilated - eroded) > 0)
    p, g    = pred[valid], gt[valid]
    tp = ((p == 1) & (g == 1)).sum()
    fp = ((p == 1) & (g == 0)).sum()
    tn = ((p == 0) & (g == 0)).sum()
    fn = ((p == 0) & (g == 1)).sum()
    prec   = tp / (tp + fp + 1e-10)
    rec    = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * rec / (prec + rec + 1e-10)
    sh_iou = tp / (tp + fp + fn + 1e-10)
    ns_iou = tn / (tn + fp + fn + 1e-10)
    miou   = (sh_iou + ns_iou) / 2
    oa     = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    sh_e   = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0.0
    ns_e   = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0.0
    ber    = (sh_e + ns_e) / 2
    return {
        'OA': float(oa * 100), 'Precision': float(prec * 100),
        'Recall': float(rec * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(sh_iou * 100),
    }


def _avg(lst):
    if not lst:
        return {k: 0.0 for k in _METRIC_KEYS}
    return {k: float(np.mean([m[k] for m in lst])) for k in _METRIC_KEYS}


def _avg_metric_dicts(dicts):
    """Average a list of metric dicts (already averaged, not per-image lists)."""
    if not dicts:
        return {k: 0.0 for k in _METRIC_KEYS}
    return {k: float(np.mean([d.get(k, 0.0) for d in dicts])) for k in _METRIC_KEYS}


# ═══════════════════════════════════════════════════════════════════════════════
# Image loading and scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _stem_map(directory):
    """Return {stem: full_path} for all images in directory."""
    m = {}
    if not os.path.isdir(directory):
        return m
    for fn in os.listdir(directory):
        if os.path.splitext(fn)[1].lower() in _IMG_EXTS:
            m[os.path.splitext(fn)[0]] = os.path.join(directory, fn)
    return m


def _count_images(directory):
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory)
               if os.path.splitext(f)[1].lower() in _IMG_EXTS)


def score_dir(pred_dir, gt_dir, img_size, tol, label='',
              restrict_stems=None):
    """
    Score all predictions in pred_dir against gt_dir.

    restrict_stems : set of stems — only these are scored to ensure a
                     fair cross-method comparison on identical images.

    Returns dict with keys:
        strict, tolerant_Kpx, n_images, strict_list, tolerant_list, stems
    or None on failure.
    """
    pred_map = _stem_map(pred_dir)
    gt_map   = _stem_map(gt_dir)
    tol_key  = f'tolerant_{tol}px'

    if not pred_map:
        print(f'  [{label}] No predictions found: {pred_dir}')
        return None
    if not gt_map:
        print(f'  [{label}] No GT masks found: {gt_dir}')
        return None

    stems = [
        s for s in sorted(pred_map)
        if s in gt_map
        and (restrict_stems is None or s in restrict_stems)
    ]

    if not stems:
        print(f'  [{label}] No matching (pred, GT) stems in {pred_dir}')
        return None

    strict_list   = []
    tolerant_list = []
    sz = (img_size, img_size)

    for stem in stems:
        p_img = cv2.imread(pred_map[stem], cv2.IMREAD_GRAYSCALE)
        g_img = cv2.imread(gt_map[stem],   cv2.IMREAD_GRAYSCALE)
        if p_img is None or g_img is None:
            print(f'  [{label}] WARNING: could not read {stem}, skipping.')
            continue
        if p_img.shape != (img_size, img_size):
            p_img = cv2.resize(p_img, sz, interpolation=cv2.INTER_NEAREST)
        if g_img.shape != (img_size, img_size):
            g_img = cv2.resize(g_img, sz, interpolation=cv2.INTER_NEAREST)
        p_bin = (p_img > 127).astype(np.uint8)
        g_bin = (g_img > 127).astype(np.uint8)
        strict_list.append(_compute_strict(p_bin, g_bin))
        tolerant_list.append(_compute_tolerant(p_bin, g_bin, tol))

    if not strict_list:
        return None

    return {
        'n_images':      len(strict_list),
        'strict':        _avg(strict_list),
        tol_key:         _avg(tolerant_list),
        'strict_list':   strict_list,
        'tolerant_list': tolerant_list,
        'stems':         stems,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline directory resolution
# ═══════════════════════════════════════════════════════════════════════════════

def get_baseline_dirs(test_img_results, model_type, city, resolution):
    """
    Return ordered list of (label, pred_dir) for all baseline methods.
    Upper Bound is always first.
    """
    dirs = [
        ('Upper Bound',
         os.path.join(test_img_results, 'upper', city, resolution, model_type, 'base')),
    ]
    for label, subfolder in _LOCO_VARIANT_DIRS.items():
        dirs.append((
            label,
            os.path.join(test_img_results, 'loco', city, resolution,
                         model_type, subfolder),
        ))
    return dirs


# ═══════════════════════════════════════════════════════════════════════════════
# SIB variant folder discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_sib_variants(sib_output_dir, model_type, city, resolution):
    """
    Auto-discover SIB variant prediction dirs for a given model/city/resolution.

    Naming patterns (from training scripts):
      mamnet  : mamnet_sib_{TAG}__loco_holdout_{city}__{res}
      oglanet : oglanet_sib_{TAG}__loco_holdout_{city}__{res}
      dinov3  : dinov3_sib_{EXP_TAG}_{FLAG_TAG}_loco_holdout_{city}_{res}_{run_idx}
                where EXP_TAG = D7, D10 etc. (from --exp_tag)
                and FLAG_TAG = haar_vib_aug etc. (from component flags)
                Old renamed folders also match: dinov3_sib_D4_haar_vib_aug_...

    Returns dict: display_tag → pred_dir (only dirs with ≥1 prediction image).
    """
    prefix = f'{model_type}_sib'

    if model_type in ('mamnet', 'oglanet'):
        glob_pattern = os.path.join(
            sib_output_dir,
            f'{prefix}_*__loco_holdout_{city}__{resolution}')
        tag_re = re.compile(
            rf'^{re.escape(prefix)}_(.+?)__loco_holdout_{re.escape(city)}')
    else:  # dinov3
        glob_pattern = os.path.join(
            sib_output_dir,
            f'{prefix}*_loco_holdout_{city}_{resolution}*')
        tag_re = re.compile(
            rf'^{re.escape(prefix)}(.+?)_loco_holdout_{re.escape(city)}')

    found = {}
    for folder in sorted(glob.glob(glob_pattern)):
        folder_name = os.path.basename(folder)
        m = tag_re.match(folder_name)
        if not m:
            continue
        raw_tag  = m.group(1)
        tag      = raw_tag.lstrip('_')          # strip leading _ from DINOv3 tags
        pred_dir = os.path.join(folder, 'predictions')
        n = _count_images(pred_dir)
        if n > 0:
            found[tag] = pred_dir
        else:
            print(f'  [discover] No predictions in {pred_dir} — skipping.')

    return found

def discover_bypass_alpha_files(sib_output_dir, model_type, city, resolution):
    """
    Find bypass_gate_alpha.json files for module-bypass experiments.
    The alpha file lives in the experiment root (not in predictions/),
    so this is separate from score_dir which only scans predictions/.
    Returns dict: experiment_tag → json_path.
    """
    prefix = f'{model_type}_sib'
    if model_type in ('mamnet', 'oglanet'):
        glob_pattern = os.path.join(
            sib_output_dir,
            f'{prefix}_*__loco_holdout_{city}__{resolution}')
        tag_re = re.compile(
            rf'^{re.escape(prefix)}_(.+?)__loco_holdout_{re.escape(city)}')
    else:
        glob_pattern = os.path.join(
            sib_output_dir,
            f'{prefix}*_loco_holdout_{city}_{resolution}*')
        tag_re = re.compile(
            rf'^{re.escape(prefix)}(.+?)_loco_holdout_{re.escape(city)}')

    found = {}
    for folder in sorted(glob.glob(glob_pattern)):
        alpha_path = os.path.join(folder, 'bypass_gate_alpha.json')
        if not os.path.isfile(alpha_path):
            continue
        m = tag_re.match(os.path.basename(folder))
        if m:
            found[m.group(1).lstrip('_')] = alpha_path
    return found

# ═══════════════════════════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════════════════════════

_LABEL_W = 36


def _row_str(label, m, highlight=False):
    vals   = ''.join(f'{m.get(k, 0):8.2f}' for k in _METRIC_KEYS)
    marker = '>>>' if highlight else '   '
    return f'{marker} {label:<{_LABEL_W}}{vals}'


def print_table(title, rows):
    """
    rows: list of (label, metric_dict, highlight_bool)
    """
    hdr   = '    ' + ' ' * _LABEL_W + ''.join(f'{k:>8}' for k in _METRIC_KEYS)
    width = len(hdr)
    print(f'\n{"=" * width}')
    print(f'  {title}')
    print(f'{"=" * width}')
    print(hdr)
    print('-' * width)
    for label, m, highlight in rows:
        print(_row_str(label, m, highlight))
    print('-' * width)


def print_recovery_ratios(sib_strict, sib_tol, ub_strict, ub_tol,
                          loco_strict, loco_tol, tol_key, variant_label):
    """Print recovery ratios R = (SIB - LOCO) / (UB - LOCO)."""
    print(f'\n  Recovery Ratios  R = ({variant_label} − LOCO_Vanilla) / (UB − LOCO_Vanilla)')
    print(f'  0 = no help, 1 = gap fully closed')
    for eval_label, sib_m, ub_m, loco_m in [
            ('Strict  ', sib_strict, ub_strict, loco_strict),
            ('Tolerant', sib_tol,    ub_tol,    loco_tol)]:
        parts = []
        for k in ['F1', 'mIOU', 'Shadow_IOU', 'BER']:
            gap = ub_m.get(k, 0) - loco_m.get(k, 0)
            rec = sib_m.get(k, 0) - loco_m.get(k, 0)
            if k == 'BER':
                gap, rec = -gap, -rec
            R = rec / gap if abs(gap) > 0.01 else float('nan')
            parts.append(f'{k}={R:.3f}')
        print(f'  {eval_label}  ' + '  '.join(parts))


def print_f1_dist(label, strict_list):
    f1s = [m['F1'] for m in strict_list]
    n   = len(f1s)
    z   = sum(1 for v in f1s if v < 0.1)
    print(f'\n  [{label}] Strict F1 ({n} imgs):  '
          f'mean={np.mean(f1s):.2f}  med={np.median(f1s):.2f}  '
          f'std={np.std(f1s):.2f}  '
          f'min={np.min(f1s):.2f}  max={np.max(f1s):.2f}  '
          f'F1<0.1: {z}/{n}')


def _print_bootstrap(sib_list, ref_list, label, n_boot=5000):
    """Paired bootstrap test for F1, mIOU, Shadow_IOU."""
    n = min(len(sib_list), len(ref_list))
    if n == 0:
        return
    print(f'\n  Bootstrap (n_boot={n_boot}, n={n}): SIB vs {label}')
    np.random.seed(42)
    for k in ['F1', 'mIOU', 'Shadow_IOU']:
        a    = np.array([m[k] for m in sib_list[:n]])
        b    = np.array([m[k] for m in ref_list[:n]])
        diff = a - b
        obs  = np.mean(diff)
        boots = np.array([
            np.mean(diff[np.random.choice(n, n, replace=True)])
            for _ in range(n_boot)])
        ci_lo = np.percentile(boots, 2.5)
        ci_hi = np.percentile(boots, 97.5)
        p = 2 * max(
            np.mean(boots <= 0) if obs >= 0 else np.mean(boots >= 0),
            1.0 / n_boot)
        p   = min(p, 1.0)
        sig = (' ***' if p < 0.001 else ' **' if p < 0.01
               else ' *' if p < 0.05 else '')
        print(f'    {k:<12} Δ={obs:+.2f}  '
              f'CI=[{ci_lo:+.2f},{ci_hi:+.2f}]  p={p:.4f}{sig}')


# ═══════════════════════════════════════════════════════════════════════════════
# Per-city evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_city(model_type, city, resolution, sib_output_dir,
              test_img_results, gt_base_dir, img_size, tol):
    """
    Evaluate all baselines + SIB variants for one model/city/resolution.

    Returns city_result dict (no per-image lists — stripped before JSON save).
    Also returns in-memory scores (with per-image lists) for bootstrap tests.
    """
    tol_key = f'tolerant_{tol}px'

    # Resolve GT mask directory
    gt_dir = None
    for candidate in [
        os.path.join(gt_base_dir, city, resolution, 'test', 'masks'),
        os.path.join(gt_base_dir, city, resolution, 'masks'),
    ]:
        if os.path.isdir(candidate):
            gt_dir = candidate
            break

    if gt_dir is None:
        print(f'\n  [{model_type}/{city}] GT dir not found under {gt_base_dir}'
              f'/{city}/{resolution}/{{test/,}}masks/')
        return None, {}

    print(f'\n{"=" * 70}')
    print(f'  {model_type.upper()} — {city} — {resolution}')
    print(f'  GT masks : {gt_dir}')
    print(f'{"=" * 70}')

    # ── Step 1: Score Upper Bound first → get reference stem set ──────────
    baseline_dirs = get_baseline_dirs(test_img_results, model_type, city, resolution)
    baseline_scores_mem = {}   # label → full score dict (with per-image lists)

    ub_label, ub_dir = baseline_dirs[0]   # always ('Upper Bound', ...)
    print(f'\n  [Baselines]')
    n_ub = _count_images(ub_dir)
    if n_ub > 0:
        ub_scores = score_dir(ub_dir, gt_dir, img_size, tol, label=ub_label)
        if ub_scores:
            baseline_scores_mem[ub_label] = ub_scores
            ub_stems = set(ub_scores['stems'])
            print(f'    ✓ {ub_label:<22} {n_ub} imgs  '
                  f'strict F1={ub_scores["strict"]["F1"]:.2f}  '
                  f'tol F1={ub_scores[tol_key]["F1"]:.2f}')
        else:
            ub_stems = None
            print(f'    ✗ {ub_label:<22} scoring failed')
    else:
        ub_stems = None
        status = 'dir missing' if not os.path.isdir(ub_dir) else 'no images'
        print(f'    ✗ {ub_label:<22} {ub_dir}  ({status})')

    # ── Step 2: Score LOCO baselines restricted to UB stems ───────────────
    for label, pred_dir in baseline_dirs[1:]:
        n = _count_images(pred_dir)
        if n > 0:
            s = score_dir(pred_dir, gt_dir, img_size, tol,
                          label=label, restrict_stems=ub_stems)
            if s:
                baseline_scores_mem[label] = s
                print(f'    ✓ {label:<22} {s["n_images"]} imgs  '
                      f'strict F1={s["strict"]["F1"]:.2f}  '
                      f'tol F1={s[tol_key]["F1"]:.2f}')
            else:
                print(f'    ✗ {label:<22} {pred_dir}  (no matching stems)')
        else:
            status = 'dir missing' if not os.path.isdir(pred_dir) else 'no images'
            print(f'    ✗ {label:<22} {pred_dir}  ({status})')

    # ── Step 3: Discover + score SIB variants ─────────────────────────────
    sib_variant_dirs = discover_sib_variants(
        sib_output_dir, model_type, city, resolution)
    sib_scores_mem = {}   # tag → full score dict (with per-image lists)

    print(f'\n  [SIB variants] Discovered {len(sib_variant_dirs)}')
    for tag, pred_dir in sorted(sib_variant_dirs.items()):
        s = score_dir(pred_dir, gt_dir, img_size, tol,
                      label=f'SIB:{tag}', restrict_stems=ub_stems)
        if s:
            sib_scores_mem[tag] = s
            print(f'    ✓ {tag:<42} {s["n_images"]} imgs  '
                  f'strict F1={s["strict"]["F1"]:.2f}  '
                  f'tol F1={s[tol_key]["F1"]:.2f}')

    if not sib_scores_mem:
        print(f'\n  WARNING: No SIB variant scores for {model_type}/{city}')

    # ── Step 4: Print comparison tables ───────────────────────────────────
    for eval_type, type_label in [
            ('strict',  'STRICT — per-image mean'),
            (tol_key,   f'TOLERANT ±{tol}px — per-image mean  [PRIMARY]')]:
        rows = []
        for label in _ORDERED_BASELINE_LABELS:
            if label in baseline_scores_mem:
                rows.append((label, baseline_scores_mem[label][eval_type], False))
        for tag in sorted(sib_scores_mem):
            rows.append((
                f'SIB — {tag}',
                sib_scores_mem[tag][eval_type],
                True))
        if rows:
            print_table(
                f'{model_type.upper()} / {city} / {resolution} — {type_label}',
                rows)

    # ── Step 5: Recovery ratios for best SIB variant ──────────────────────
    if (sib_scores_mem
            and 'Upper Bound' in baseline_scores_mem
            and 'LOCO Vanilla' in baseline_scores_mem):
        best_tag = max(sib_scores_mem,
                       key=lambda t: sib_scores_mem[t][tol_key]['mIOU'])
        print(f'\n  Best SIB for {city}: {best_tag}  '
              f'(tol mIOU={sib_scores_mem[best_tag][tol_key]["mIOU"]:.2f})')
        print_recovery_ratios(
            sib_scores_mem[best_tag]['strict'],
            sib_scores_mem[best_tag][tol_key],
            baseline_scores_mem['Upper Bound']['strict'],
            baseline_scores_mem['Upper Bound'][tol_key],
            baseline_scores_mem['LOCO Vanilla']['strict'],
            baseline_scores_mem['LOCO Vanilla'][tol_key],
            tol_key, best_tag)

    # ── Step 6: Per-image F1 distributions ────────────────────────────────
    for label in ['Upper Bound', 'LOCO Vanilla']:
        if label in baseline_scores_mem:
            print_f1_dist(label, baseline_scores_mem[label]['strict_list'])
    for tag in sorted(sib_scores_mem):
        print_f1_dist(f'SIB:{tag}', sib_scores_mem[tag]['strict_list'])

    # ── Step 7: Bootstrap tests (best SIB vs each LOCO baseline) ──────────
    if sib_scores_mem:
        best_tag = max(sib_scores_mem,
                       key=lambda t: sib_scores_mem[t][tol_key]['mIOU'])
        for bl_label in ['LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
                         'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']:
            if bl_label in baseline_scores_mem:
                print(f'\n  {"─"*65}')
                _print_bootstrap(
                    sib_scores_mem[best_tag]['strict_list'],
                    baseline_scores_mem[bl_label]['strict_list'],
                    f'{bl_label} (strict)')
                _print_bootstrap(
                    sib_scores_mem[best_tag]['tolerant_list'],
                    baseline_scores_mem[bl_label]['tolerant_list'],
                    f'{bl_label} (tolerant)')

    # ── Build output dict (no per-image lists) ────────────────────────────
    city_result = {
        'baselines':    {},
        'sib_variants': {},
    }
    for label, s in baseline_scores_mem.items():
        city_result['baselines'][label] = {
            'strict':   s['strict'],
            tol_key:    s[tol_key],
            'n_images': s['n_images'],
        }
    for tag, s in sib_scores_mem.items():
        city_result['sib_variants'][tag] = {
            'strict':   s['strict'],
            tol_key:    s[tol_key],
            'n_images': s['n_images'],
        }

    return city_result, sib_scores_mem


# ═══════════════════════════════════════════════════════════════════════════════
# Variant ranking across cities
# ═══════════════════════════════════════════════════════════════════════════════

def rank_variants(cities_results, tol_key, cities):
    """
    Rank SIB variants by average tolerant mIOU across cities.

    Returns (best_tag, ranked_list).
    ranked_list: list of dicts sorted best-first.
    """
    # Collect all tags seen across cities
    all_tags = set()
    for city in cities:
        cr = cities_results.get(city)
        if cr:
            all_tags |= set(cr.get('sib_variants', {}).keys())

    if not all_tags:
        return None, []

    variant_stats = {}
    for tag in all_tags:
        tol_mious    = []
        strict_mious = []
        cities_won   = 0

        for city in cities:
            cr = cities_results.get(city)
            if not cr:
                continue
            sv = cr.get('sib_variants', {})
            if tag not in sv:
                continue
            tol_mious.append(sv[tag][tol_key]['mIOU'])
            strict_mious.append(sv[tag]['strict']['mIOU'])

            # Did this tag win this city?
            if sv:
                city_winner = max(sv, key=lambda t: sv[t][tol_key]['mIOU'])
                if city_winner == tag:
                    cities_won += 1

        variant_stats[tag] = {
            'avg_tolerant_miou': float(np.mean(tol_mious))    if tol_mious    else 0.0,
            'med_tolerant_miou': float(np.median(tol_mious))  if tol_mious    else 0.0,
            'avg_strict_miou':   float(np.mean(strict_mious)) if strict_mious else 0.0,
            'cities_won':        cities_won,
            'n_cities_scored':   len(tol_mious),
            'per_city': {
                city: cities_results[city]['sib_variants'][tag][tol_key]['mIOU']
                for city in cities
                if cities_results.get(city)
                and tag in cities_results[city].get('sib_variants', {})
            },
        }

    ranked = sorted(
        variant_stats.items(),
        key=lambda x: (x[1]['avg_tolerant_miou'], x[1]['cities_won']),
        reverse=True)

    best_tag = ranked[0][0]

    print(f'\n{"=" * 70}')
    print(f'  SIB VARIANT RANKING — avg tolerant mIOU across {len(cities)} cities')
    print(f'{"=" * 70}')
    print(f'  {"Rank":<5} {"Tag":<45} {"AvgTol":>8} {"MedTol":>8} '
          f'{"AvgStr":>8} {"CityWon":>8}')
    print('  ' + '-' * 68)
    for rank, (tag, stats) in enumerate(ranked, 1):
        marker = ' ★' if rank == 1 else '  '
        print(f'  {rank:<5}{marker}{tag:<43} '
              f'{stats["avg_tolerant_miou"]:8.2f} '
              f'{stats["med_tolerant_miou"]:8.2f} '
              f'{stats["avg_strict_miou"]:8.2f} '
              f'{stats["cities_won"]:8d}')

    ranked_list = [{'tag': tag, **stats} for tag, stats in ranked]
    return best_tag, ranked_list


# ═══════════════════════════════════════════════════════════════════════════════
# Round 2 focused analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _find_tag_by_prefix(tags, prefix):
    """Find the first tag that starts with `prefix_` or equals `prefix` exactly."""
    for tag in sorted(tags):
        if tag == prefix or tag.startswith(prefix + '_'):
            return tag
    return None


def _get_variant_miou(cities_results, tag, city, tol_key):
    """Get tolerant mIOU for a SIB variant in a city. Returns None if missing."""
    cr = cities_results.get(city)
    if not cr:
        return None
    sv = cr.get('sib_variants', {}).get(tag)
    if not sv:
        return None
    return sv.get(tol_key, {}).get('mIOU')


def _get_variant_metric(cities_results, tag, city, tol_key, metric_key):
    """Get a specific tolerant metric for a SIB variant."""
    cr = cities_results.get(city)
    if not cr:
        return None
    sv = cr.get('sib_variants', {}).get(tag)
    if not sv:
        return None
    return sv.get(tol_key, {}).get(metric_key)


def _get_baseline_miou(cities_results, city, label, tol_key):
    """Get tolerant mIOU for a baseline method in a city."""
    cr = cities_results.get(city)
    if not cr:
        return None
    bl = cr.get('baselines', {}).get(label)
    if not bl:
        return None
    return bl.get(tol_key, {}).get('mIOU')


def round2_analysis(model_type, cities_results, tol_key, cities):
    """
    Print focused analysis of Round 2 experiments:
      1. Tolerant mIOU overview table (baselines + R1 context + R2 new)
      2. Pairwise comparison deltas (each question answered by Δ per city)
      3. Multi-metric detail for each R2 variant (F1, BER, Shadow_IOU)
      4. Decision summary
    """
    config = _ROUND2_CONFIG.get(model_type)
    if not config:
        return

    # Collect all discovered tags across cities
    all_tags = set()
    for city in cities:
        cr = cities_results.get(city)
        if cr:
            all_tags |= set(cr.get('sib_variants', {}).keys())

    if not all_tags:
        return

    # Map experiment prefixes → actual discovered tags
    all_prefixes = config['round2_prefixes'] + config['context_prefixes']
    prefix_to_tag = {}
    for prefix in all_prefixes:
        tag = _find_tag_by_prefix(all_tags, prefix)
        if tag:
            prefix_to_tag[prefix] = tag

    found_r2 = [p for p in config['round2_prefixes'] if p in prefix_to_tag]
    if not found_r2:
        print(f'\n  [Round 2] No Round 2 variants found for {model_type}')
        return

    COL_W  = 12
    LBL_W  = 48
    n_cols = len(cities) + 1   # cities + avg
    sep_w  = LBL_W + COL_W * n_cols

    # ── Table 1: Tolerant mIOU overview ───────────────────────────────────
    print(f'\n{"=" * sep_w}')
    print(f'  ROUND 2 ANALYSIS — {model_type.upper()} — Tolerant mIOU ({tol_key})')
    print(f'{"=" * sep_w}')

    hdr = f'  {"Variant":<{LBL_W}}'
    for c in cities:
        hdr += f'{c:>{COL_W}}'
    hdr += f'{"Avg":>{COL_W}}'
    print(hdr)
    print(f'  ' + '-' * (sep_w - 2))

    def _print_miou_row(label, values, marker='  '):
        """Print a row of mIOU values with average."""
        vals_str = ''
        valid = []
        for v in values:
            if v is not None:
                vals_str += f'{v:{COL_W}.2f}'
                valid.append(v)
            else:
                vals_str += f'{"—":>{COL_W}}'
        avg = np.mean(valid) if valid else 0
        avg_str = f'{avg:{COL_W}.2f}' if valid else f'{"—":>{COL_W}}'
        print(f'{marker}{label:<{LBL_W}}{vals_str}{avg_str}')

    # Baselines row
    for bl_label in ['Upper Bound', 'LOCO Vanilla']:
        vals = [_get_baseline_miou(cities_results, c, bl_label, tol_key)
                for c in cities]
        _print_miou_row(bl_label, vals)

    # Best non-SIB baseline per city
    best_bl_vals = []
    for c in cities:
        cr = cities_results.get(c)
        if not cr:
            best_bl_vals.append(None)
            continue
        best_v = 0
        for bl_label in ['LOCO FDA', 'LOCO SegDesic', 'LOCO IIM',
                         'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']:
            v = _get_baseline_miou(cities_results, c, bl_label, tol_key)
            if v is not None and v > best_v:
                best_v = v
        best_bl_vals.append(best_v if best_v > 0 else None)
    _print_miou_row('Best Adapt. Baseline', best_bl_vals)

    print(f'  ' + '-' * (sep_w - 2))

    # Round 1 context variants
    for prefix in config['context_prefixes']:
        tag = prefix_to_tag.get(prefix)
        if not tag:
            continue
        vals = [_get_variant_miou(cities_results, tag, c, tol_key)
                for c in cities]
        display = f'R1: {prefix} ({tag})'
        if len(display) > LBL_W:
            display = f'R1: {prefix} ({tag[:LBL_W - len(prefix) - 8]})'
        _print_miou_row(display, vals)

    print(f'  ' + '-' * (sep_w - 2))

    # Round 2 variants
    for prefix in config['round2_prefixes']:
        tag = prefix_to_tag.get(prefix)
        if not tag:
            print(f'  {">>":>2} {prefix:<{LBL_W - 2}}{"(not found)":>{COL_W}}')
            continue
        vals = [_get_variant_miou(cities_results, tag, c, tol_key)
                for c in cities]
        display = f'R2: {prefix} ({tag})'
        if len(display) > LBL_W:
            display = f'R2: {prefix} ({tag[:LBL_W - len(prefix) - 8]})'
        _print_miou_row(display, vals, marker='>>')

    # ── Table 2: Pairwise deltas ──────────────────────────────────────────
    print(f'\n{"=" * sep_w}')
    print(f'  ROUND 2 PAIRWISE — {model_type.upper()} — Δ Tolerant mIOU (A − B)')
    print(f'  Positive = A is better')
    print(f'{"=" * sep_w}')

    pair_lbl_w = 20
    q_w = 35
    hdr2 = f'  {"A vs B":<{pair_lbl_w}} {"Question":<{q_w}}'
    for c in cities:
        hdr2 += f'{c:>{COL_W}}'
    hdr2 += f'{"Avg":>{COL_W}}'
    print(hdr2)
    print(f'  ' + '-' * (pair_lbl_w + q_w + COL_W * n_cols))

    for prefix_a, prefix_b, question in config['pairs']:
        tag_a = prefix_to_tag.get(prefix_a)
        tag_b = prefix_to_tag.get(prefix_b)
        if not tag_a or not tag_b:
            pair_label = f'{prefix_a} vs {prefix_b}'
            print(f'  {pair_label:<{pair_lbl_w}} {"MISSING DATA":<{q_w}}')
            continue

        deltas = []
        delta_strs = ''
        for c in cities:
            va = _get_variant_miou(cities_results, tag_a, c, tol_key)
            vb = _get_variant_miou(cities_results, tag_b, c, tol_key)
            if va is not None and vb is not None:
                d = va - vb
                deltas.append(d)
                delta_strs += f'{d:+{COL_W}.2f}'
            else:
                delta_strs += f'{"—":>{COL_W}}'

        avg_d = np.mean(deltas) if deltas else 0
        avg_str = f'{avg_d:+{COL_W}.2f}' if deltas else f'{"—":>{COL_W}}'

        pair_label = f'{prefix_a} vs {prefix_b}'
        q_display = question[:q_w - 1]
        print(f'  {pair_label:<{pair_lbl_w}} {q_display:<{q_w}}{delta_strs}{avg_str}')

    # ── Table 3: Multi-metric detail for R2 variants ──────────────────────
    detail_metrics = ['mIOU', 'F1', 'BER', 'Shadow_IOU']
    print(f'\n{"=" * sep_w}')
    print(f'  ROUND 2 MULTI-METRIC — {model_type.upper()} — Tolerant ({tol_key})')
    print(f'{"=" * sep_w}')

    for prefix in config['round2_prefixes']:
        tag = prefix_to_tag.get(prefix)
        if not tag:
            continue
        print(f'\n  {prefix} ({tag}):')
        met_hdr = f'    {"Metric":<14}'
        for c in cities:
            met_hdr += f'{c:>{COL_W}}'
        met_hdr += f'{"Avg":>{COL_W}}'
        print(met_hdr)
        print(f'    ' + '-' * (14 + COL_W * n_cols))

        for mk in detail_metrics:
            vals = []
            for c in cities:
                v = _get_variant_metric(cities_results, tag, c, tol_key, mk)
                vals.append(v)
            valid = [v for v in vals if v is not None]
            avg = np.mean(valid) if valid else 0

            val_str = ''
            for v in vals:
                if v is not None:
                    val_str += f'{v:{COL_W}.2f}'
                else:
                    val_str += f'{"—":>{COL_W}}'
            avg_str = f'{avg:{COL_W}.2f}' if valid else f'{"—":>{COL_W}}'
            print(f'    {mk:<14}{val_str}{avg_str}')

    # ── Decision summary ──────────────────────────────────────────────────
    print(f'\n{"=" * sep_w}')
    print(f'  ROUND 2 DECISION SUMMARY — {model_type.upper()}')
    print(f'{"=" * sep_w}')

    for prefix_a, prefix_b, question in config['pairs']:
        tag_a = prefix_to_tag.get(prefix_a)
        tag_b = prefix_to_tag.get(prefix_b)
        if not tag_a or not tag_b:
            print(f'  {prefix_a} vs {prefix_b}: MISSING — '
                  f'{"tag_a=" + str(tag_a) + " tag_b=" + str(tag_b)}')
            continue

        wins = 0
        losses = 0
        ties = 0
        for c in cities:
            va = _get_variant_miou(cities_results, tag_a, c, tol_key)
            vb = _get_variant_miou(cities_results, tag_b, c, tol_key)
            if va is not None and vb is not None:
                if va > vb + 0.1:
                    wins += 1
                elif vb > va + 0.1:
                    losses += 1
                else:
                    ties += 1

        total = wins + losses + ties
        if wins > losses:
            verdict = f'{prefix_a} WINS ({wins}W/{ties}T/{losses}L across {total} cities)'
        elif losses > wins:
            verdict = f'{prefix_b} WINS ({losses}W/{ties}T/{wins}L across {total} cities)'
        else:
            verdict = f'TIE ({wins}W/{ties}T/{losses}L across {total} cities)'

        print(f'  {prefix_a} vs {prefix_b}')
        print(f'    Q: {question}')
        print(f'    → {verdict}')

    # ── Best R2 variant overall ───────────────────────────────────────────
    r2_scores = {}
    for prefix in config['round2_prefixes']:
        tag = prefix_to_tag.get(prefix)
        if not tag:
            continue
        vals = [_get_variant_miou(cities_results, tag, c, tol_key)
                for c in cities]
        valid = [v for v in vals if v is not None]
        if valid:
            r2_scores[prefix] = (tag, np.mean(valid))

    if r2_scores:
        best_r2_prefix = max(r2_scores, key=lambda p: r2_scores[p][1])
        best_r2_tag, best_r2_avg = r2_scores[best_r2_prefix]
        print(f'\n  ★ Best Round 2 variant: {best_r2_prefix} ({best_r2_tag})')
        print(f'    Avg tolerant mIOU = {best_r2_avg:.2f}')

        # Compare to best R1 context
        for prefix in config['context_prefixes']:
            tag = prefix_to_tag.get(prefix)
            if not tag:
                continue
            vals = [_get_variant_miou(cities_results, tag, c, tol_key)
                    for c in cities]
            valid = [v for v in vals if v is not None]
            if valid:
                ctx_avg = np.mean(valid)
                delta = best_r2_avg - ctx_avg
                print(f'    vs R1 {prefix} ({tag}): '
                      f'Δ = {delta:+.2f} mIOU')

    return prefix_to_tag


# ═══════════════════════════════════════════════════════════════════════════════
# Unified candidate assessment
# ═══════════════════════════════════════════════════════════════════════════════

def unified_candidate_assessment(cities_results, tol_key, cities, best_tag):
    """
    For the overall best SIB variant, show how it compares to every
    individual baseline across all cities.  Reports whether the variant
    beats each baseline in all / majority / minority of cells.
    """
    if not best_tag:
        return

    print(f'\n{"=" * 80}')
    print(f'  UNIFIED CANDIDATE ASSESSMENT — Best overall: {best_tag}')
    print(f'{"=" * 80}')

    COL_W = 12
    LBL_W = 22

    hdr = f'  {"Baseline":<{LBL_W}}'
    for c in cities:
        hdr += f'{c:>{COL_W}}'
    hdr += f'{"Avg Δ":>{COL_W}}  {"Verdict"}'
    print(hdr)
    print(f'  ' + '-' * (LBL_W + COL_W * (len(cities) + 1) + 12))

    bl_labels = ['LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
                 'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']

    for bl_label in bl_labels:
        deltas = []
        delta_strs = ''
        for c in cities:
            sib_v = _get_variant_miou(cities_results, best_tag, c, tol_key)
            bl_v  = _get_baseline_miou(cities_results, c, bl_label, tol_key)
            if sib_v is not None and bl_v is not None:
                d = sib_v - bl_v
                deltas.append(d)
                delta_strs += f'{d:+{COL_W}.2f}'
            else:
                delta_strs += f'{"—":>{COL_W}}'

        if deltas:
            avg_d = np.mean(deltas)
            wins = sum(1 for d in deltas if d > 0.1)
            ties = sum(1 for d in deltas if abs(d) <= 0.1)
            losses = sum(1 for d in deltas if d < -0.1)
            if wins == len(deltas):
                verdict = f'✓ WINS ALL ({len(deltas)}/{len(deltas)})'
            elif losses == 0:
                verdict = f'~ NO LOSS ({wins}W {ties}T)'
            else:
                verdict = f'✗ {wins}W {ties}T {losses}L'
            avg_str = f'{avg_d:+{COL_W}.2f}'
        else:
            avg_str = f'{"—":>{COL_W}}'
            verdict = 'N/A'

        print(f'  {bl_label:<{LBL_W}}{delta_strs}{avg_str}  {verdict}')

    # Worst-case degradation
    all_deltas_vs_vanilla = []
    for c in cities:
        sib_v = _get_variant_miou(cities_results, best_tag, c, tol_key)
        van_v = _get_baseline_miou(cities_results, c, 'LOCO Vanilla', tol_key)
        if sib_v is not None and van_v is not None:
            all_deltas_vs_vanilla.append(sib_v - van_v)

    if all_deltas_vs_vanilla:
        worst = min(all_deltas_vs_vanilla)
        print(f'\n  Worst-case vs Vanilla: {worst:+.2f} mIOU')
        if worst >= -0.5:
            print(f'  → No catastrophic failure (worst > −0.5)')
        else:
            print(f'  → WARNING: degradation > 0.5 mIOU in at least one city')


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-model comparison
# ═══════════════════════════════════════════════════════════════════════════════

def cross_model_comparison(output_dir, resolution, tol_key, cities):
    """
    If eval JSONs for all 3 models exist, print and save cross-model tables.
    Auto-triggered at the end of each model's eval run.
    """
    eval_files = {
        m: os.path.join(output_dir, f'eval_{m}_{resolution}.json')
        for m in _MODEL_TYPES
    }
    missing = [m for m, p in eval_files.items() if not os.path.isfile(p)]
    if missing:
        print(f'\n  [Cross-model] Not all evals complete — missing: {missing}')
        print(f'  Cross-model table will be printed when all 3 are done.')
        return

    print(f'\n{"=" * 70}')
    print(f'  CROSS-MODEL COMPARISON  (resolution={resolution}  ±{tol_key})')
    print(f'{"=" * 70}')

    model_data = {}
    for m, path in eval_files.items():
        with open(path) as f:
            model_data[m] = json.load(f)

    # Best variant per model (from summary)
    best_per_model = {}
    for m, data in model_data.items():
        summary  = data.get('summary', {})
        best_tag = summary.get('best_variant', 'N/A')
        avg_tol  = summary.get('best_variant_avg_tolerant_miou', 0.0)
        best_per_model[m] = {'variant': best_tag, 'avg_tolerant_miou': avg_tol}
        print(f'  {m.upper():<10} best variant: {best_tag}  '
              f'(avg tol mIOU={avg_tol:.2f})')

    # Per-city tables
    per_city_out = {}
    for city in cities:
        per_city_out[city] = {}
        rows_tol    = []
        rows_strict = []

        # Upper Bound and LOCO Vanilla rows (from mamnet, as reference)
        ref_city = model_data['mamnet'].get('cities', {}).get(city, {})
        for label in ['Upper Bound', 'LOCO Vanilla']:
            bl = ref_city.get('baselines', {}).get(label)
            if bl:
                rows_tol.append(    (label, bl[tol_key], False))
                rows_strict.append( (label, bl['strict'], False))

        # Best SIB per model
        for m in _MODEL_TYPES:
            best_tag  = best_per_model[m]['variant']
            city_data = model_data[m].get('cities', {}).get(city, {})
            sv        = city_data.get('sib_variants', {})
            if best_tag in sv:
                s   = sv[best_tag]
                lbl = f'{m.upper()} — {best_tag[:28]}'
                rows_tol.append(   (lbl, s[tol_key], True))
                rows_strict.append((lbl, s['strict'],  True))
                per_city_out[city][m] = {
                    'variant': best_tag,
                    tol_key:   s[tol_key],
                    'strict':  s['strict'],
                }

        if rows_tol:
            print_table(
                f'CROSS-MODEL / {city} / TOLERANT ±{tol_key}', rows_tol)
        if rows_strict:
            print_table(
                f'CROSS-MODEL / {city} / STRICT', rows_strict)

    # Average-across-cities table
    print(f'\n{"=" * 70}')
    print(f'  CROSS-MODEL / AVG ACROSS {len(cities)} CITIES / TOLERANT ±{tol_key}')
    print(f'{"=" * 70}')

    avg_rows = []
    # UB and LOCO Vanilla avg rows
    for label in ['Upper Bound', 'LOCO Vanilla']:
        city_ms = []
        for city in cities:
            ref_city = model_data['mamnet'].get('cities', {}).get(city, {})
            bl = ref_city.get('baselines', {}).get(label)
            if bl:
                city_ms.append(bl[tol_key])
        if city_ms:
            avg_rows.append((f'{label} (avg)', _avg_metric_dicts(city_ms), False))

    for m in _MODEL_TYPES:
        best_tag  = best_per_model[m]['variant']
        city_ms   = []
        for city in cities:
            city_data = model_data[m].get('cities', {}).get(city, {})
            sv        = city_data.get('sib_variants', {})
            if best_tag in sv:
                city_ms.append(sv[best_tag][tol_key])
        if city_ms:
            avg_rows.append((
                f'{m.upper()} SIB — {best_tag[:28]} (avg)',
                _avg_metric_dicts(city_ms),
                True))

    if avg_rows:
        print_table('AVG ACROSS CITIES — TOLERANT', avg_rows)

    # Save cross-model JSON
    cross_out = {
        'resolution':     resolution,
        'tol_key':        tol_key,
        'cities':         cities,
        'best_per_model': best_per_model,
        'per_city':       per_city_out,
    }
    out_path = os.path.join(
        output_dir, f'cross_model_comparison_{resolution}.json')
    with open(out_path, 'w') as f:
        json.dump(cross_out, f, indent=4)
    print(f'\n  Cross-model comparison saved → {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Unified SIB evaluation for MAMNet / OGLANet / DINOv3')
    p.add_argument('--model_type', required=True,
                   choices=_MODEL_TYPES,
                   help='Model architecture to evaluate')
    p.add_argument('--sib_output_dir', required=True,
                   help='Model-specific SIB outputs dir, e.g. '
                        '.../data/mamnet/outputs/')
    p.add_argument('--test_img_results_dir', required=True,
                   help='Root of Test_img_results/ containing baseline preds')
    p.add_argument('--gt_base_dir', required=True,
                   help='Root of Final_data_test/')
    p.add_argument('--output_dir', required=True,
                   help='Where to save eval JSON files')
    p.add_argument('--resolution', type=str, default='highres',
                   choices=['highres', 'midres'])
    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help='±K px don\'t-care zone for tolerant metrics')
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--cities', nargs='+',
                   default=['chicago', 'miami', 'phoenix'])
    return p.parse_args()

def bypass_gate_analysis(model_type, cities, sib_output_dir, resolution,
                         cities_results, tol_key):
    """
    Analyse module-level bypass gate experiments (M14 / O13 / D11).

    Key questions:
      1. Does gate α correlate positively with domain gap?
         (high gap city → α near 1; negligible gap city → α near 0)
      2. Does bypass fix regressions that C4 introduced?
      3. Are C4 gains preserved where they existed?
    """
    bypass_prefixes = {'mamnet': ['M14'], 'oglanet': ['O13'], 'dinov3': ['D11']}
    c4_prefix       = {'mamnet': 'M1',   'oglanet': 'M1',    'dinov3': 'D1'}
    prior_best      = {'mamnet': 'M1',   'oglanet': 'M1',    'dinov3': 'D8'}

    prefixes = bypass_prefixes.get(model_type, [])
    all_tags = set()
    for city in cities:
        cr = cities_results.get(city)
        if cr:
            all_tags |= set(cr.get('sib_variants', {}).keys())

    bypass_tags = {p: _find_tag_by_prefix(all_tags, p)
                   for p in prefixes
                   if _find_tag_by_prefix(all_tags, p)}

    if not bypass_tags:
        print(f'\n  [Bypass Gate Analysis] No bypass variants found — skipping.')
        return

    c4_tag   = _find_tag_by_prefix(all_tags, c4_prefix[model_type])
    best_tag = _find_tag_by_prefix(all_tags, prior_best[model_type])

    print(f'\n{"#" * 70}')
    print(f'  MODULE BYPASS GATE ANALYSIS — {model_type.upper()}')
    print(f'  Did the gate learn to modulate SIB strength by domain gap?')
    print(f'{"#" * 70}')

    for prefix, bypass_tag in bypass_tags.items():

        # ── 1. Alpha values per city ──────────────────────────────────
        print(f'\n  [{prefix}]  Gate α per city  (α=1 → full SIB,  α=0 → bypass)')
        print(f'  {"─" * 65}')
        alpha_per_city = {}
        for city in cities:
            afiles = discover_bypass_alpha_files(
                sib_output_dir, model_type, city, resolution)
            apath = afiles.get(bypass_tag)
            if not apath:
                print(f'    {city:<12}  bypass_gate_alpha.json not found')
                continue
            with open(apath) as f:
                ad = json.load(f)
            ma = ad.get('mean_alpha',   ad.get('mean',   float('nan')))
            sa = ad.get('std_alpha',    ad.get('std',    float('nan')))
            pa = ad.get('median_alpha', ad.get('median', float('nan')))
            lo = ad.get('min_alpha',    ad.get('min',    float('nan')))
            hi = ad.get('max_alpha',    ad.get('max',    float('nan')))
            ni = ad.get('num_images',   ad.get('n_images', '?'))
            alpha_per_city[city] = ma
            print(f'    {city:<12}  mean={ma:.3f}  std={sa:.3f}  '
                  f'median={pa:.3f}  range=[{lo:.3f},{hi:.3f}]  n={ni}')

        # ── 2. α vs domain gap ────────────────────────────────────────
        if len(alpha_per_city) >= 2:
            print(f'\n  Domain gap (UB − Vanilla tol mIOU) vs mean α:')
            print(f'  {"City":<12} {"UB":>8} {"Vanilla":>9} {"Gap":>7} '
                  f'{"α":>8}  {"Expected behaviour"}')
            print(f'  {"─" * 65}')
            gaps, alphas = [], []
            for city in cities:
                cr   = cities_results.get(city, {})
                ub_v = cr.get('baselines', {}).get('Upper Bound',   {}).get(tol_key, {}).get('mIOU')
                vn_v = cr.get('baselines', {}).get('LOCO Vanilla',  {}).get(tol_key, {}).get('mIOU')
                a    = alpha_per_city.get(city)
                if None in (ub_v, vn_v, a):
                    continue
                gap = ub_v - vn_v
                exp = ('α → 0 (bypass)' if gap < 5 else
                       'α → 0.5 (partial)' if gap < 15 else 'α → 1 (apply SIB)')
                print(f'  {city:<12} {ub_v:8.2f} {vn_v:9.2f} {gap:7.2f} {a:8.3f}  {exp}')
                gaps.append(gap); alphas.append(a)

            if len(gaps) >= 2:
                r = np.corrcoef(gaps, alphas)[0, 1]
                verdict = ('✓ gate α rises with domain gap' if r > 0.5 else
                           '✗ inverse — gate misaligned' if r < -0.3 else
                           '~ weak correlation')
                print(f'\n  Pearson r(gap, α) = {r:.3f}  →  {verdict}')

        # ── 3. Performance table ──────────────────────────────────────
        print(f'\n  Performance comparison (tolerant mIOU):')
        CW = 10
        hdr = f'  {"City":<12}{"Vanilla":>{CW}}'
        if c4_tag:  hdr += f'{"C4 base":>{CW}}'
        if best_tag and best_tag != c4_tag: hdr += f'{"BestPrior":>{CW}}'
        hdr += f'{"Bypass":>{CW}}'
        if c4_tag:  hdr += f'{"Δ vs C4":>{CW}}'
        hdr += f'{"Δ vs Van":>{CW}}'
        print(hdr)
        print(f'  {"─" * 65}')

        byp_vals, c4_vals, van_vals = [], [], []
        for city in cities:
            cr  = cities_results.get(city, {})
            vn  = cr.get('baselines', {}).get('LOCO Vanilla', {}).get(tol_key, {}).get('mIOU')
            c4  = _get_variant_miou(cities_results, c4_tag,     city, tol_key) if c4_tag else None
            bp  = _get_variant_miou(cities_results, bypass_tag, city, tol_key)
            bst = _get_variant_miou(cities_results, best_tag,   city, tol_key) if best_tag and best_tag != c4_tag else None
            if vn  is not None: van_vals.append(vn)
            if c4  is not None: c4_vals.append(c4)
            if bp  is not None: byp_vals.append(bp)
            row = f'  {city:<12}{vn:{CW}.2f}' if vn is not None else f'  {city:<12}{"—":>{CW}}'
            if c4_tag:   row += f'{c4:{CW}.2f}' if c4 is not None else f'{"—":>{CW}}'
            if bst is not None: row += f'{bst:{CW}.2f}'
            elif best_tag and best_tag != c4_tag: row += f'{"—":>{CW}}'
            row += f'{bp:{CW}.2f}' if bp is not None else f'{"—":>{CW}}'
            if c4_tag:   row += (f'{bp-c4:+{CW}.2f}' if None not in (bp, c4) else f'{"—":>{CW}}')
            row += (f'{bp-vn:+{CW}.2f}' if None not in (bp, vn) else f'{"—":>{CW}}')
            print(row)

        if byp_vals:
            print(f'  {"─" * 65}')
            avg_v = np.mean(van_vals) if van_vals else None
            avg_c = np.mean(c4_vals)  if c4_vals  else None
            avg_b = np.mean(byp_vals)
            row   = f'  {"Avg":<12}{avg_v:{CW}.2f}' if avg_v else f'  {"Avg":<12}{"—":>{CW}}'
            if c4_tag:   row += f'{avg_c:{CW}.2f}' if avg_c else f'{"—":>{CW}}'
            if best_tag and best_tag != c4_tag: row += f'{"":>{CW}}'
            row += f'{avg_b:{CW}.2f}'
            if c4_tag:   row += (f'{avg_b-avg_c:+{CW}.2f}' if avg_c else f'{"—":>{CW}}')
            row += (f'{avg_b-avg_v:+{CW}.2f}' if avg_v else f'{"—":>{CW}}')
            print(row)

        # ── 4. Verdict ────────────────────────────────────────────────
        print(f'\n  VERDICT — {prefix}:')
        regressions = [(c, _get_variant_miou(cities_results, c4_tag, c, tol_key),
                           cities_results.get(c, {}).get('baselines', {})
                                .get('LOCO Vanilla', {}).get(tol_key, {}).get('mIOU'))
                       for c in cities if c4_tag]
        reg_cities   = [c for c, c4v, vnv in regressions
                        if None not in (c4v, vnv) and c4v < vnv - 0.5]
        fixed_cities = [c for c in reg_cities
                        if (_get_variant_miou(cities_results, bypass_tag, c, tol_key) or 0) >=
                           (cities_results.get(c, {}).get('baselines', {})
                                .get('LOCO Vanilla', {}).get(tol_key, {}).get('mIOU') or 0) - 0.5]
        if reg_cities:
            print(f'    C4 regressions vs vanilla: {reg_cities}')
            print(f'    Fixed by bypass: {fixed_cities}  '
                  + ('✓' if fixed_cities == reg_cities else '✗'))
        else:
            print(f'    C4 had no regressions — bypass should preserve all gains')

        n_win  = sum(1 for c in cities
                     if (_get_variant_miou(cities_results, bypass_tag, c, tol_key) or 0) >
                        (cities_results.get(c,{}).get('baselines',{})
                             .get('LOCO Vanilla',{}).get(tol_key,{}).get('mIOU') or 0) + 0.1)
        n_loss = sum(1 for c in cities
                     if (_get_variant_miou(cities_results, bypass_tag, c, tol_key) or 0) < 
                        (cities_results.get(c,{}).get('baselines',{})
                             .get('LOCO Vanilla',{}).get(tol_key,{}).get('mIOU') or 0) - 0.5)
        avg_b = np.mean(byp_vals) if byp_vals else 0
        avg_v = np.mean(van_vals) if van_vals else 0
        print(f'    vs Vanilla: {n_win}W / {n_loss}L across {len(cities)} cities')
        if n_loss == 0 and avg_b > avg_v:
            print(f'    → ADOPT: no regressions, positive average gain '
                  f'(+{avg_b - avg_v:.2f} mIOU)')
        elif n_loss == 0:
            print(f'    → NEUTRAL: no regressions but marginal average gain')
        else:
            print(f'    → REJECT for universal use: {n_loss} regression(s) remain')

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    tol     = args.boundary_tolerance
    tol_key = f'tolerant_{tol}px'

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'  EVAL: {args.model_type.upper()}  '
          f'resolution={args.resolution}  tol=±{tol}px')
    print(f'  SIB outputs dir    : {args.sib_output_dir}')
    print(f'  Test img results   : {args.test_img_results_dir}')
    print(f'  GT base dir        : {args.gt_base_dir}')
    print(f'  Output dir         : {args.output_dir}')
    print(f'  Cities             : {args.cities}')
    print(f'{"=" * 70}')

    # Evaluate each city
    cities_results = {}  # city → city_result dict (no per-image lists)
    for city in args.cities:
        city_result, _ = eval_city(
            model_type=args.model_type,
            city=city,
            resolution=args.resolution,
            sib_output_dir=args.sib_output_dir,
            test_img_results=args.test_img_results_dir,
            gt_base_dir=args.gt_base_dir,
            img_size=args.img_size,
            tol=tol,
        )
        cities_results[city] = city_result

    # Rank variants across cities
    best_tag, ranked = rank_variants(cities_results, tol_key, args.cities)

    # ── Round 2 focused analysis ──────────────────────────────────────────
    r2_prefix_map = round2_analysis(
        args.model_type, cities_results, tol_key, args.cities)

    # ── Unified candidate assessment ──────────────────────────────────────
    if best_tag:
        unified_candidate_assessment(
            cities_results, tol_key, args.cities, best_tag)
        
    bypass_gate_analysis(
        args.model_type, args.cities, args.sib_output_dir,
        args.resolution, cities_results, tol_key)

    # Build + save per-model eval JSON
    output = {
        'model_type':         args.model_type,
        'resolution':         args.resolution,
        'boundary_tolerance': tol,
        'img_size':           args.img_size,
        'cities':             {c: r for c, r in cities_results.items() if r},
        'summary': {
            'best_variant':                   best_tag,
            'best_variant_avg_tolerant_miou': ranked[0]['avg_tolerant_miou'] if ranked else 0.0,
            'variants_ranked':                ranked,
        },
    }

    # Include Round 2 tag mapping in JSON for downstream scripts
    if r2_prefix_map:
        output['round2'] = {
            'prefix_to_tag': r2_prefix_map,
            'config':        _ROUND2_CONFIG.get(args.model_type, {}),
        }

    out_path = os.path.join(
        args.output_dir,
        f'eval_{args.model_type}_{args.resolution}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f'\n  Saved → {out_path}')

    # Cross-model comparison (auto-triggers when all 3 model JSONs exist)
    cross_model_comparison(
        args.output_dir, args.resolution, tol_key, args.cities)

    print(f'\nDone!')


if __name__ == '__main__':
    main()