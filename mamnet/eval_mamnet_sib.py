"""
Evaluation script for MAMNet+SIB.

Two evaluation modes:
  --mode predictions  : Score saved prediction PNGs directly against GT masks.
  --mode checkpoint   : Reload best_model.pth and run fresh inference.

Both modes report THREE aggregation methods side-by-side for SIB AND
upper bound (when --ub_pred_dir is supplied), so you can see exactly
which method produces which number and compare apples-to-apples:

  Aggregation A  Strict  dataset-level concat  (old method — all pixels pooled)
  Aggregation B  Strict  per-image mean        (new method — every image equal weight)
  Aggregation C  Tolerant ±5px per-image mean

The final comparison table prints all methods x all aggregations in one view.
Upper bound predictions are always restricted to the same image stems as SIB
so the comparison is fair regardless of which images each directory contains.

Usage examples:
  # Score saved PNGs + upper bound
  python eval_mamnet_sib.py \
      --mode predictions \
      --pred_dir    .../mamnet_sib_M2_.../predictions/ \
      --gt_dir      .../Final_data_test/miami/highres/test/masks/ \
      --ub_pred_dir .../Test_img_results/upper/miami/highres/mamnet/base/ \
      --img_size 384

  # Fresh inference from checkpoint + upper bound
  python eval_mamnet_sib.py \
      --mode checkpoint \
      --ckpt_path   .../mamnet_sib_M2_.../best_model.pth \
      --data_root   .../Final_data_test/miami/highres/ \
      --ub_pred_dir .../Test_img_results/upper/miami/highres/mamnet/base/ \
      --img_size 384 --batch_size 4

  # Also add LOCO vanilla to get recovery ratios
  python eval_mamnet_sib.py \
      --mode predictions \
      --pred_dir     .../predictions/ \
      --gt_dir       .../test/masks/ \
      --ub_pred_dir  .../upper/miami/highres/mamnet/base/ \
      --loco_pred_dir .../loco/miami/highres/mamnet/vanilla/ \
      --img_size 384
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════

def compute_strict(pred, gt):
    """pred, gt: flat or 2-D uint8/bool arrays with values 0/1."""
    pred = pred.flatten().astype(bool)
    gt   = gt.flatten().astype(bool)
    tp = ( pred &  gt).sum()
    fp = ( pred & ~gt).sum()
    tn = (~pred & ~gt).sum()
    fn = (~pred &  gt).sum()
    prec   = tp / (tp + fp + 1e-10)
    rec    = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * rec / (prec + rec + 1e-10)
    sh_iou = tp / (tp + fp + fn + 1e-10)
    ns_iou = tn / (tn + fp + fn + 1e-10)
    miou   = (sh_iou + ns_iou) / 2
    oa     = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    sh_err = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0.0
    ns_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0.0
    ber    = (sh_err + ns_err) / 2
    return dict(OA=float(oa*100), Precision=float(prec*100),
                Recall=float(rec*100), F1=float(f1*100),
                BER=float(ber*100), mIOU=float(miou*100),
                Shadow_IOU=float(sh_iou*100))


_KERN_CACHE = {}
def _kern(tol):
    if tol not in _KERN_CACHE:
        _KERN_CACHE[tol] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tol*2+1, tol*2+1))
    return _KERN_CACHE[tol]


def compute_tolerant(pred, gt, tol=5):
    """pred, gt: 2-D uint8 arrays with values 0/1."""
    k       = _kern(tol)
    gt_u8   = gt.astype(np.uint8)
    eroded  = cv2.erode(gt_u8, k)
    dilated = cv2.dilate(gt_u8, k)
    valid   = ~((dilated - eroded) > 0)
    p, g    = pred[valid], gt[valid]
    tp = ((p==1) & (g==1)).sum()
    fp = ((p==1) & (g==0)).sum()
    tn = ((p==0) & (g==0)).sum()
    fn = ((p==0) & (g==1)).sum()
    prec   = tp / (tp + fp + 1e-10)
    rec    = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * rec / (prec + rec + 1e-10)
    sh_iou = tp / (tp + fp + fn + 1e-10)
    ns_iou = tn / (tn + fp + fn + 1e-10)
    miou   = (sh_iou + ns_iou) / 2
    oa     = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    sh_err = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0.0
    ns_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0.0
    ber    = (sh_err + ns_err) / 2
    return dict(OA=float(oa*100), Precision=float(prec*100),
                Recall=float(rec*100), F1=float(f1*100),
                BER=float(ber*100), mIOU=float(miou*100),
                Shadow_IOU=float(sh_iou*100))


def average_metrics(lst):
    if not lst:
        return {k: 0.0 for k in
                ['OA','Precision','Recall','F1','BER','mIOU','Shadow_IOU']}
    keys = ['OA','Precision','Recall','F1','BER','mIOU','Shadow_IOU']
    return {k: float(np.mean([m[k] for m in lst])) for k in keys}


def dataset_level_metrics(all_preds, all_gts):
    """Concatenate all pixels then compute once — old method."""
    p = np.concatenate([x.flatten() for x in all_preds])
    g = np.concatenate([x.flatten() for x in all_gts])
    return compute_strict(p, g)


# ══════════════════════════════════════════════════════════════════════
# Printing helpers
# ══════════════════════════════════════════════════════════════════════

_METRIC_KEYS = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
_COL_W = 42   # label column width


def _row_str(label, m, bold=False):
    vals = ''.join(f'{m.get(k, 0):8.2f}' for k in _METRIC_KEYS)
    marker = '>>>' if bold else '   '
    return f'{marker} {label:<{_COL_W}}{vals}'


def print_comparison_table(title, rows):
    """
    rows: list of (label, metric_dict, is_bold)
    """
    hdr = '    ' + ' '*_COL_W + ''.join(f'{k:>8}' for k in _METRIC_KEYS)
    width = len(hdr)
    print(f'\n{"="*width}')
    print(f'  {title}')
    print(f'{"="*width}')
    print(hdr)
    print('-'*width)
    for item in rows:
        label, m = item[0], item[1]
        bold = item[2] if len(item) > 2 else False
        print(_row_str(label, m, bold=bold))
    print('-'*width)


def print_f1_distribution(label, strict_list):
    f1s = [m['F1'] for m in strict_list]
    n   = len(f1s)
    zero_f1 = sum(1 for v in f1s if v < 0.1)
    print(f'\n  [{label}] Per-image strict F1 distribution ({n} images):')
    print(f'    mean={np.mean(f1s):.2f}  median={np.median(f1s):.2f}  '
          f'std={np.std(f1s):.2f}  min={np.min(f1s):.2f}  max={np.max(f1s):.2f}')
    print(f'    Images with F1 < 0.1: {zero_f1}/{n} '
          f'({100*zero_f1/max(n,1):.1f}%)  <- these drag the per-image mean down')


# ══════════════════════════════════════════════════════════════════════
# Print all aggregations in one unified view
# ══════════════════════════════════════════════════════════════════════

def print_full_comparison(named_scores):
    """
    named_scores: ordered list of (name, scores_dict, is_sib)
      scores_dict must have keys: strict_dataset_level, strict_per_image_mean,
                                  tolerant_5px_per_image
    """
    aggregations = [
        ('strict_dataset_level',
         'A — Strict, dataset-level concat  (old method — all pixels pooled)'),
        ('strict_per_image_mean',
         'B — Strict, per-image mean        (new method — equal image weight)'),
        ('tolerant_5px_per_image',
         'C — Tolerant +/-5px, per-image mean'),
    ]

    for agg_key, agg_title in aggregations:
        rows = []
        for name, scores, is_sib in named_scores:
            rows.append((name, scores[agg_key], is_sib))

        # Delta rows: SIB minus each reference
        sib_m = next((s[agg_key] for n, s, b in named_scores if b), None)
        if sib_m is not None:
            for name, scores, is_sib in named_scores:
                if not is_sib:
                    delta = {k: sib_m[k] - scores[agg_key][k]
                             for k in _METRIC_KEYS}
                    rows.append((f'  delta  SIB - {name}', delta, False))

        print_comparison_table(agg_title, rows)


def print_recovery_table(sib_scores, ub_scores, loco_scores):
    """R = (SIB - LOCO) / (UB - LOCO) for each aggregation x metric."""
    aggregations = [
        ('strict_dataset_level',   'A  dataset-level  '),
        ('strict_per_image_mean',  'B  per-image mean '),
        ('tolerant_5px_per_image', 'C  tolerant +-5px '),
    ]
    print('\n  ── Recovery Ratios  R = (SIB - LOCO) / (UB - LOCO) ──')
    for agg_key, agg_label in aggregations:
        sib  = sib_scores[agg_key]
        ub   = ub_scores[agg_key]
        loco = loco_scores[agg_key]
        parts = []
        for k in ['F1', 'mIOU', 'Shadow_IOU', 'BER']:
            gap = ub[k] - loco[k]
            rec = sib[k] - loco[k]
            if k == 'BER':
                gap, rec = -gap, -rec
            R = rec / gap if abs(gap) > 0.01 else float('nan')
            parts.append(f'{k}={R:.3f}')
        print(f'  {agg_label}  ' + '  '.join(parts))


# ══════════════════════════════════════════════════════════════════════
# Core scoring utility — shared between both modes
# ══════════════════════════════════════════════════════════════════════

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _stem_map(directory):
    m = {}
    if not os.path.isdir(directory):
        return m
    for fn in os.listdir(directory):
        if os.path.splitext(fn)[1].lower() in IMG_EXTS:
            m[os.path.splitext(fn)[0]] = os.path.join(directory, fn)
    return m


def score_pred_dir(pred_dir, gt_dir, img_size, label='',
                   restrict_stems=None):
    """
    Score prediction PNGs in pred_dir against GT masks in gt_dir.

    restrict_stems: if provided (a set of stems), only these are scored —
                    ensures fair comparison across methods.

    Returns dict with all three aggregation methods plus raw per-image lists,
    or None if scoring fails.
    """
    pred_map = _stem_map(pred_dir)
    gt_map   = _stem_map(gt_dir)

    if not pred_map:
        print(f'  [{label}] WARNING: no images in pred_dir: {pred_dir}')
        return None
    if not gt_map:
        print(f'  [{label}] WARNING: no images in gt_dir:   {gt_dir}')
        return None

    if restrict_stems is not None:
        stems = [s for s in sorted(pred_map)
                 if s in gt_map and s in restrict_stems]
    else:
        stems = [s for s in sorted(pred_map) if s in gt_map]

    restriction_note = (f', restricted to {len(restrict_stems)} reference stems'
                        if restrict_stems else '')
    print(f'\n  [{label}] Matched {len(stems)} pairs '
          f'(pred={len(pred_map)}, gt={len(gt_map)}{restriction_note})')

    if not stems:
        print(f'  [{label}] ERROR: no matching stems found.')
        return None

    sz            = img_size
    strict_list   = []
    tolerant_list = []
    all_preds     = []
    all_gts       = []

    for stem in stems:
        pred_img = cv2.imread(pred_map[stem], cv2.IMREAD_GRAYSCALE)
        gt_img   = cv2.imread(gt_map[stem],   cv2.IMREAD_GRAYSCALE)
        if pred_img is None or gt_img is None:
            print(f'  [{label}] WARNING: could not read {stem}, skipping.')
            continue
        if pred_img.shape != (sz, sz):
            pred_img = cv2.resize(pred_img, (sz, sz),
                                  interpolation=cv2.INTER_NEAREST)
        if gt_img.shape != (sz, sz):
            gt_img   = cv2.resize(gt_img,   (sz, sz),
                                  interpolation=cv2.INTER_NEAREST)
        pred_bin = (pred_img > 127).astype(np.uint8)
        gt_bin   = (gt_img   > 127).astype(np.uint8)

        strict_list.append(compute_strict(pred_bin, gt_bin))
        tolerant_list.append(compute_tolerant(pred_bin, gt_bin, tol=5))
        all_preds.append(pred_bin)
        all_gts.append(gt_bin)

    n = len(strict_list)
    if n == 0:
        return None

    return {
        'n_images':               n,
        'strict_dataset_level':   dataset_level_metrics(all_preds, all_gts),
        'strict_per_image_mean':  average_metrics(strict_list),
        'tolerant_5px_per_image': average_metrics(tolerant_list),
        'strict_list':            strict_list,
        'tolerant_list':          tolerant_list,
        'all_preds':              all_preds,
        'all_gts':                all_gts,
        'stems':                  stems,
    }


def score_arrays(all_preds, all_gts, stems=None, label=''):
    """Score pre-loaded numpy arrays (from checkpoint inference)."""
    strict_list   = [compute_strict(p, g)   for p, g in zip(all_preds, all_gts)]
    tolerant_list = [compute_tolerant(p, g) for p, g in zip(all_preds, all_gts)]
    return {
        'n_images':               len(all_preds),
        'strict_dataset_level':   dataset_level_metrics(all_preds, all_gts),
        'strict_per_image_mean':  average_metrics(strict_list),
        'tolerant_5px_per_image': average_metrics(tolerant_list),
        'strict_list':            strict_list,
        'tolerant_list':          tolerant_list,
        'all_preds':              all_preds,
        'all_gts':                all_gts,
        'stems':                  stems or [],
    }


def _per_image_gap_summary(sib_scores, ref_scores, ref_label):
    sib_f1s = np.array([m['F1'] for m in sib_scores['strict_list']])
    ref_f1s = np.array([m['F1'] for m in ref_scores['strict_list']])
    n = min(len(sib_f1s), len(ref_f1s))
    if n == 0:
        return
    deltas = sib_f1s[:n] - ref_f1s[:n]
    print(f'\n  Per-image strict F1 delta  (SIB - {ref_label})  over {n} images:')
    print(f'    mean={np.mean(deltas):+.2f}  median={np.median(deltas):+.2f}  '
          f'std={np.std(deltas):.2f}')
    print(f'    Images where SIB is >5pt worse  than {ref_label}: '
          f'{(deltas < -5).sum()}')
    print(f'    Images where SIB is >5pt better than {ref_label}: '
          f'{(deltas > 5).sum()}')


def _build_output_dict(sib_scores, ub_scores, loco_scores):
    out = {
        'n_images_sib': sib_scores['n_images'],
        'sib': {
            'strict_dataset_level':   sib_scores['strict_dataset_level'],
            'strict_per_image_mean':  sib_scores['strict_per_image_mean'],
            'tolerant_5px_per_image': sib_scores['tolerant_5px_per_image'],
        },
    }
    if ub_scores is not None:
        out['n_images_upper_bound'] = ub_scores['n_images']
        out['upper_bound'] = {
            'strict_dataset_level':   ub_scores['strict_dataset_level'],
            'strict_per_image_mean':  ub_scores['strict_per_image_mean'],
            'tolerant_5px_per_image': ub_scores['tolerant_5px_per_image'],
        }
    if loco_scores is not None:
        out['n_images_loco_vanilla'] = loco_scores['n_images']
        out['loco_vanilla'] = {
            'strict_dataset_level':   loco_scores['strict_dataset_level'],
            'strict_per_image_mean':  loco_scores['strict_per_image_mean'],
            'tolerant_5px_per_image': loco_scores['tolerant_5px_per_image'],
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# MODE 1: Score saved prediction PNGs
# ══════════════════════════════════════════════════════════════════════

def eval_predictions(args):
    print(f'\n{"="*65}')
    print(f'MODE: Score saved prediction PNGs')
    print(f'  pred_dir     : {args.pred_dir}')
    print(f'  gt_dir       : {args.gt_dir}')
    print(f'  ub_pred_dir  : {args.ub_pred_dir  or "(not provided)"}')
    print(f'  loco_pred_dir: {args.loco_pred_dir or "(not provided)"}')
    print(f'  img_size     : {args.img_size}')
    print(f'{"="*65}')

    # Score SIB
    sib_scores = score_pred_dir(
        args.pred_dir, args.gt_dir, args.img_size, label='SIB')
    if sib_scores is None:
        print('FATAL: could not score SIB predictions.')
        return None
    sib_stems = set(sib_scores['stems'])

    # Score upper bound — restricted to exactly the same stems as SIB
    ub_scores = None
    if args.ub_pred_dir:
        print(f'\n  Scoring Upper Bound (same {len(sib_stems)} stems as SIB)...')
        ub_scores = score_pred_dir(
            args.ub_pred_dir, args.gt_dir, args.img_size,
            label='Upper Bound', restrict_stems=sib_stems)
        if ub_scores is None:
            print('  WARNING: upper bound scoring failed.')
    else:
        print('\n  No --ub_pred_dir provided; upper bound skipped.')

    # Score LOCO vanilla (optional, for recovery ratios)
    loco_scores = None
    if args.loco_pred_dir:
        print(f'\n  Scoring LOCO-vanilla (same stems)...')
        loco_scores = score_pred_dir(
            args.loco_pred_dir, args.gt_dir, args.img_size,
            label='LOCO Vanilla', restrict_stems=sib_stems)

    # Build ordered list for unified table
    named_scores = [('SIB (this run)', sib_scores, True)]
    if ub_scores   is not None: named_scores.append(('Upper Bound',  ub_scores,   False))
    if loco_scores is not None: named_scores.append(('LOCO Vanilla', loco_scores, False))

    print_full_comparison(named_scores)

    if ub_scores is not None and loco_scores is not None:
        print_recovery_table(sib_scores, ub_scores, loco_scores)

    # Per-image distributions
    print_f1_distribution('SIB', sib_scores['strict_list'])
    if ub_scores   is not None: print_f1_distribution('Upper Bound',  ub_scores['strict_list'])
    if loco_scores is not None: print_f1_distribution('LOCO Vanilla', loco_scores['strict_list'])

    if ub_scores   is not None: _per_image_gap_summary(sib_scores, ub_scores,   'Upper Bound')
    if loco_scores is not None: _per_image_gap_summary(sib_scores, loco_scores, 'LOCO Vanilla')

    # Save
    out = _build_output_dict(sib_scores, ub_scores, loco_scores)
    out_path = os.path.normpath(
        os.path.join(args.pred_dir, '..', 'eval_results.json'))
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=4)
    print(f'\n  Saved -> {out_path}')
    return out


# ══════════════════════════════════════════════════════════════════════
# MODE 2: Fresh inference from checkpoint
# ══════════════════════════════════════════════════════════════════════

def eval_checkpoint(args):
    print(f'\n{"="*65}')
    print(f'MODE: Fresh inference from checkpoint')
    print(f'  ckpt_path    : {args.ckpt_path}')
    print(f'  data_root    : {args.data_root}')
    print(f'  ub_pred_dir  : {args.ub_pred_dir  or "(not provided)"}')
    print(f'  loco_pred_dir: {args.loco_pred_dir or "(not provided)"}')
    print(f'  img_size     : {args.img_size}')
    print(f'{"="*65}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args_dict = ckpt.get('args', {})
    print(f'  Checkpoint epoch {ckpt.get("epoch","?")}  '
          f'best_metric={ckpt.get("best_metric","?"):.4f}')
    print('  Training args saved in checkpoint:')
    for k, v in sorted(train_args_dict.items()):
        print(f'    {k} = {v}')

    import argparse as _ap
    saved = _ap.Namespace(**train_args_dict)
    saved.img_size  = args.img_size
    saved.data_root = args.data_root

    from models.mamnet_sib import build_mamnet_sib
    model = build_mamnet_sib(saved).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print('  Model loaded.')

    from data.dataset_sib import ShadowDatasetSIB
    from torch.utils.data import DataLoader

    test_ds = ShadowDatasetSIB(
        root_dir=[args.data_root], split='test',
        img_size=args.img_size, augment=False,
        use_contrast=train_args_dict.get('use_contrast', False),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    print(f'  Test dataset: {len(test_ds)} images')

    from utils.postprocessing import filter_small_predictions

    all_preds    = []
    all_gts      = []
    all_stems    = []

    with torch.no_grad():
        for batch in test_loader:
            images        = batch['image'].to(device)
            masks         = batch['mask'].to(device)
            intensity_map = batch['intensity_map'].to(device)
            filenames     = batch['filename']

            outputs, _ = model(images, intensity_map=intensity_map)
            logits = (outputs if isinstance(outputs, torch.Tensor)
                      else outputs['main'])

            filtered = filter_small_predictions(logits, min_pixels=10)
            preds    = filtered.argmax(dim=1)

            for i in range(preds.shape[0]):
                all_preds.append(preds[i].cpu().numpy().astype(np.uint8))
                all_gts.append(masks[i].cpu().numpy().astype(np.uint8))
                all_stems.append(os.path.splitext(filenames[i])[0])

    # Score SIB inference
    sib_scores = score_arrays(all_preds, all_gts, stems=all_stems, label='SIB')
    sib_stems  = set(all_stems)

    # Resolve GT dir for upper bound / loco scoring
    gt_dir = os.path.join(args.data_root, 'test', 'masks')
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(args.data_root, 'masks')
    print(f'\n  GT dir: {gt_dir}')

    # Score upper bound
    ub_scores = None
    if args.ub_pred_dir:
        print(f'\n  Scoring Upper Bound (restricted to {len(sib_stems)} SIB stems)...')
        ub_scores = score_pred_dir(
            args.ub_pred_dir, gt_dir, args.img_size,
            label='Upper Bound', restrict_stems=sib_stems)
        if ub_scores is None:
            print('  WARNING: upper bound scoring failed.')
    else:
        print('\n  No --ub_pred_dir provided; upper bound skipped.')

    # Score LOCO vanilla
    loco_scores = None
    if args.loco_pred_dir:
        print(f'\n  Scoring LOCO-vanilla (restricted to same stems)...')
        loco_scores = score_pred_dir(
            args.loco_pred_dir, gt_dir, args.img_size,
            label='LOCO Vanilla', restrict_stems=sib_stems)

    # Unified table
    named_scores = [('SIB (this run)', sib_scores, True)]
    if ub_scores   is not None: named_scores.append(('Upper Bound',  ub_scores,   False))
    if loco_scores is not None: named_scores.append(('LOCO Vanilla', loco_scores, False))

    print_full_comparison(named_scores)

    if ub_scores is not None and loco_scores is not None:
        print_recovery_table(sib_scores, ub_scores, loco_scores)

    print_f1_distribution('SIB', sib_scores['strict_list'])
    if ub_scores   is not None: print_f1_distribution('Upper Bound',  ub_scores['strict_list'])
    if loco_scores is not None: print_f1_distribution('LOCO Vanilla', loco_scores['strict_list'])

    if ub_scores   is not None: _per_image_gap_summary(sib_scores, ub_scores,   'Upper Bound')
    if loco_scores is not None: _per_image_gap_summary(sib_scores, loco_scores, 'LOCO Vanilla')

    out = _build_output_dict(sib_scores, ub_scores, loco_scores)
    out_path = os.path.join(os.path.dirname(args.ckpt_path),
                            'eval_results_checkpoint.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=4)
    print(f'\n  Saved -> {out_path}')
    return out


# ══════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate MAMNet+SIB with upper bound comparison')
    p.add_argument('--mode', required=True,
                   choices=['predictions', 'checkpoint'])

    # predictions mode
    p.add_argument('--pred_dir',  type=str, default=None,
                   help='[predictions] Directory of saved SIB prediction PNGs')
    p.add_argument('--gt_dir',    type=str, default=None,
                   help='[predictions] Directory of GT mask PNGs')

    # checkpoint mode
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='[checkpoint] Path to best_model.pth')
    p.add_argument('--data_root', type=str, default=None,
                   help='[checkpoint] Data root e.g. .../miami/highres/')
    p.add_argument('--batch_size', type=int, default=4)

    # upper bound — works in both modes
    p.add_argument('--ub_pred_dir', type=str, default=None,
                   help='Upper Bound prediction PNGs directory. '
                        'Typical path: '
                        '.../Test_img_results/upper/{city}/{res}/mamnet/base/')

    # LOCO vanilla — optional, used for recovery ratios
    p.add_argument('--loco_pred_dir', type=str, default=None,
                   help='[optional] LOCO-vanilla predictions for recovery ratios. '
                        'Typical path: '
                        '.../Test_img_results/loco/{city}/{res}/mamnet/vanilla/')

    # shared
    p.add_argument('--img_size', type=int, default=384)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'predictions':
        if not args.pred_dir or not args.gt_dir:
            print('ERROR: --pred_dir and --gt_dir required for predictions mode')
            sys.exit(1)
        eval_predictions(args)
    else:
        if not args.ckpt_path or not args.data_root:
            print('ERROR: --ckpt_path and --data_root required for checkpoint mode')
            sys.exit(1)
        eval_checkpoint(args)