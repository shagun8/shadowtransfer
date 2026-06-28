"""
Visualization utilities for MAMNet training.

plot_loss_curves  — overview panel + one subplot per loss component.
                    Each component subplot has its own y-axis scale so
                    small values are visible without compression.
                    Total loss is shown ONLY in the overview panel.
                    Supports segdesic domain losses via optional args.

plot_metrics_curves — per-metric train/val curves across epochs.

save_best_worst_visualizations — qualitative prediction samples.
                    Handles both tensor and dict model outputs safely.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
import cv2

# --------------------------------------------------------------------------
# Global rcParams for publication-quality output
# --------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.titlesize':  13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# --------------------------------------------------------------------------
# Colour palette
# --------------------------------------------------------------------------
_C_TRAIN_TOTAL  = '#2E86AB'
_C_VAL_TOTAL    = '#A23B72'
_C_MAIN_TRAIN   = '#F18F01'
_C_MAIN_VAL     = '#C73E1D'
_C_AUX_TRAIN    = '#6A994E'
_C_DOM_SRC      = '#023E8A'   # dark blue  — source domain loss
_C_DOM_TGT      = '#FF6B35'   # orange     — target domain loss

_METRIC_COLORS = {
    'OA':         '#2E86AB',
    'Precision':  '#F18F01',
    'F1':         '#A23B72',
    'BER':        '#E63946',
    'mIOU':       '#6A994E',
    'Shadow_IOU': '#8338EC',
}

MAX_COLS = 4


def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def _is_nonzero(lst):
    """Return True if the list exists and contains at least one non-trivially-zero value."""
    return lst is not None and len(lst) > 0 and any(abs(v) > 1e-9 for v in lst)


# --------------------------------------------------------------------------
# Loss curves
# --------------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses, save_path,
                     train_main_losses=None, train_aux_losses=None,
                     train_domain_src_losses=None, train_domain_tgt_losses=None):
    """
    Generate and save a comprehensive loss figure.

    Layout
    ------
    Row 0 (full-width overview):
        All available losses on a *shared* y-axis so their relative
        magnitudes are visible at a glance.  Total loss IS shown here.

    Row 1+ (individual component panels, one per available component):
        Each panel shows a *single* loss component with its own y-axis so
        fine-grained decreases invisible in the overview become clear.
        Total loss is NOT shown in individual panels.

        Panels generated when data is available and non-zero:
          • Main (CE) Loss     — train_main + val (val is main CE in eval mode)
          • Auxiliary Loss     — train_aux only
          • Domain Source Loss — train_domain_src only (SegDesic)
          • Domain Target Loss — train_domain_tgt only (SegDesic)

    Args
    ----
    train_losses            : list[float]  total training loss per epoch (required)
    val_losses              : list[float]  validation (main CE) loss per epoch (required)
    save_path               : str          output PNG path
    train_main_losses       : list[float]  main CE train loss per epoch (optional)
    train_aux_losses        : list[float]  weighted aux train loss per epoch (optional)
    train_domain_src_losses : list[float]  raw source domain loss per epoch (optional, SegDesic)
    train_domain_tgt_losses : list[float]  raw target domain loss per epoch (optional, SegDesic)
    """
    epochs = list(range(1, len(train_losses) + 1))

    # ---- Decide which individual panels to generate (no total loss panel) ----
    individual_panels = []

    if _is_nonzero(train_main_losses) and len(train_main_losses) == len(epochs):
        individual_panels.append({
            'title': 'Main (CE) Loss',
            'ylabel': 'Loss',
            'series': [
                (train_main_losses, 'Train Main CE', _C_MAIN_TRAIN, '-',  'o'),
                (val_losses,        'Val (Main CE)', _C_MAIN_VAL,   '--', 's'),
            ],
        })

    if _is_nonzero(train_aux_losses) and len(train_aux_losses) == len(epochs):
        individual_panels.append({
            'title': 'Auxiliary Loss (weighted)',
            'ylabel': 'Loss',
            'series': [
                (train_aux_losses, 'Train Aux', _C_AUX_TRAIN, '-', '^'),
            ],
        })

    if _is_nonzero(train_domain_src_losses) and len(train_domain_src_losses) == len(epochs):
        individual_panels.append({
            'title': 'Domain Source Loss (raw)',
            'ylabel': 'Cosine Dissimilarity',
            'series': [
                (train_domain_src_losses, 'Train Domain Src', _C_DOM_SRC, '-', 'D'),
            ],
        })

    if _is_nonzero(train_domain_tgt_losses) and len(train_domain_tgt_losses) == len(epochs):
        individual_panels.append({
            'title': 'Domain Target Loss (raw)',
            'ylabel': 'Cosine Dissimilarity',
            'series': [
                (train_domain_tgt_losses, 'Train Domain Tgt', _C_DOM_TGT, '-', 'P'),
            ],
        })

    n_ind      = len(individual_panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind > 0 else 0

    fig_w = max(9, min(5.5 * max(n_ind, 1), 22))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind > 0:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.55,
            height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Row 0: Overview (all losses, shared y-axis, includes total) ----
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses, '--', lw=1.8, color=_C_VAL_TOTAL,
               label='Val (Main CE)', marker='s', ms=3, mfc='white', mew=1.2)
    if _is_nonzero(train_main_losses) and len(train_main_losses) == len(epochs):
        ax_ov.plot(epochs, train_main_losses, '-', lw=1.5, color=_C_MAIN_TRAIN,
                   alpha=0.75, label='Train Main CE')
    if _is_nonzero(train_aux_losses) and len(train_aux_losses) == len(epochs):
        ax_ov.plot(epochs, train_aux_losses, '-', lw=1.5, color=_C_AUX_TRAIN,
                   alpha=0.75, label='Train Aux (weighted)')
    if _is_nonzero(train_domain_src_losses) and len(train_domain_src_losses) == len(epochs):
        ax_ov.plot(epochs, train_domain_src_losses, '-', lw=1.5, color=_C_DOM_SRC,
                   alpha=0.80, label='Train Domain Src (raw)')
    if _is_nonzero(train_domain_tgt_losses) and len(train_domain_tgt_losses) == len(epochs):
        ax_ov.plot(epochs, train_domain_tgt_losses, '-', lw=1.5, color=_C_DOM_TGT,
                   alpha=0.80, label='Train Domain Tgt (raw)')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss / Dissimilarity')
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=min(4, 2 + n_ind))
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Rows 1+: Individual component panels (own y-scale, no total) ----
    if n_ind > 0:
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_ind, MAX_COLS, subplot_spec=outer[1],
            hspace=0.55, wspace=0.42)

        for pidx, panel in enumerate(individual_panels):
            r, c = divmod(pidx, MAX_COLS)
            ax   = fig.add_subplot(inner[r, c])

            for vals, lbl, col, ls, mk in panel['series']:
                ep = list(range(1, len(vals) + 1))
                ax.plot(ep, vals, ls=ls, lw=1.8, color=col, label=lbl,
                        marker=mk, ms=3.5, mfc='white', mew=1.2, alpha=0.9)

            _style_ax(ax, panel['title'], panel['ylabel'])

        # Hide unused subplot slots
        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('MAMNet — Training Loss Curves', fontweight='bold',
                 fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


# --------------------------------------------------------------------------
# Metric curves
# --------------------------------------------------------------------------

def plot_metrics_curves(train_metrics_history, val_metrics_history, save_path):
    """
    Plot one subplot per metric, each showing train and val across epochs.

    Args
    ----
    train_metrics_history : dict {metric_name: list[float per epoch]}
    val_metrics_history   : dict {metric_name: list[float per epoch]}
    save_path             : str
    """
    metrics = list(train_metrics_history.keys())
    n       = len(metrics)
    if n == 0:
        return

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 4.0 * n_rows),
                              squeeze=False)

    for idx, metric in enumerate(metrics):
        r, c    = divmod(idx, n_cols)
        ax      = axes[r][c]
        color   = _METRIC_COLORS.get(metric, '#333333')
        vals_tr = train_metrics_history[metric]
        vals_vl = val_metrics_history[metric]
        ep_tr   = list(range(1, len(vals_tr) + 1))
        ep_vl   = list(range(1, len(vals_vl) + 1))

        ax.plot(ep_tr, vals_tr, '-', lw=1.8, color=color,
                label='Train', marker='o', ms=3.5, mfc='white', mew=1.2)
        ax.plot(ep_vl, vals_vl, '--', lw=1.8, color=color,
                label='Val',   marker='s', ms=3.5, mfc='white', mew=1.2,
                alpha=0.75)
        _style_ax(ax, metric, '%')

    # Hide unused subplots
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle('MAMNet — Metric Curves (pooled ShadowMetrics, reference)',
                 fontweight='bold', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Metric curves saved → {save_path}')


# --------------------------------------------------------------------------
# Best / worst visualizations
# --------------------------------------------------------------------------

def save_best_worst_visualizations(model, dataloader, device, output_dir,
                                    num_images=10):
    """
    Save the best and worst predicted samples ranked by per-image IoU.

    Each row in the output grid: Input | GT mask | Predicted mask | Overlay

    Handles both tensor and dict model outputs safely — MAMNetSegDesic
    returns a dict {'main': ..., 'aux1': ..., ...} while base MAMNet
    returns a raw tensor in eval mode. Both are handled.

    Args
    ----
    model        : trained model (set to eval mode internally)
    dataloader   : test DataLoader
    device       : torch.device
    output_dir   : directory to save images
    num_images   : how many best + worst samples to save (each)
    """
    model.eval()
    results = []   # list of (iou, image_np, gt_np, pred_np)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            raw_outputs = model(images)

            # Safely extract the main logit tensor regardless of output type
            if isinstance(raw_outputs, dict):
                logits = raw_outputs['main']
            else:
                logits = raw_outputs

            preds = torch.argmax(logits, dim=1)

            for i in range(images.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_np   = masks[i].cpu().numpy().astype(np.uint8)

                tp  = float(np.logical_and(pred_np == 1, gt_np == 1).sum())
                fp  = float(np.logical_and(pred_np == 1, gt_np == 0).sum())
                fn  = float(np.logical_and(pred_np == 0, gt_np == 1).sum())
                iou = tp / (tp + fp + fn + 1e-7)

                # Denormalise image for display (handle 3 or 4 channels)
                img_t = images[i].cpu().numpy()           # [C, H, W]
                img_t = np.transpose(img_t, (1, 2, 0))   # [H, W, C]
                if img_t.shape[2] >= 3:
                    img_rgb = img_t[:, :, :3] * std + mean
                    img_rgb = np.clip(img_rgb, 0, 1)
                else:
                    img_rgb = np.repeat(img_t[:, :, :1], 3, axis=2)

                results.append((iou, img_rgb, gt_np, pred_np))

    if not results:
        return

    results.sort(key=lambda x: x[0])
    worst = results[:num_images]
    best  = results[-num_images:]

    for label, subset in [('worst', worst), ('best', best)]:
        n         = len(subset)
        fig, axes = plt.subplots(n, 4, figsize=(16, 3.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, (iou, img_rgb, gt_np, pred_np) in enumerate(subset):
            # Overlay: green = TP, red = FP, blue = FN
            overlay = img_rgb.copy()
            tp_mask = np.logical_and(pred_np == 1, gt_np == 1)
            fp_mask = np.logical_and(pred_np == 1, gt_np == 0)
            fn_mask = np.logical_and(pred_np == 0, gt_np == 1)
            overlay[tp_mask] = overlay[tp_mask] * 0.4 + np.array([0, 0.8, 0]) * 0.6
            overlay[fp_mask] = overlay[fp_mask] * 0.4 + np.array([0.9, 0, 0]) * 0.6
            overlay[fn_mask] = overlay[fn_mask] * 0.4 + np.array([0, 0, 0.9]) * 0.6

            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title(f'Input  (IoU={iou:.3f})', fontsize=9)
            axes[row, 1].imshow(gt_np,   cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title('GT Mask', fontsize=9)
            axes[row, 2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title('Predicted', fontsize=9)
            axes[row, 3].imshow(np.clip(overlay, 0, 1))
            axes[row, 3].set_title('Overlay (G=TP R=FP B=FN)', fontsize=9)

            for ax in axes[row]:
                ax.axis('off')

        fig.suptitle(f'MAMNet — {label.capitalize()} Predictions', fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{label}_predictions.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'{label.capitalize()} predictions saved → {out_path}')