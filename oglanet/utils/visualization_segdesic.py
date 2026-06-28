"""
Visualization utilities for OGLANet training.

plot_loss_curves  — overview panel (all losses on shared y-axis) +
                    one subplot per COMPONENT loss on its own y-axis scale.
                    Total loss appears ONLY in the overview panel.
                    Individual panels show each component at its own scale
                    so small decreases that would be invisible in the overview
                    are clearly visible.  Val total loss is also shown ONLY
                    in the overview panel — it is NOT repeated as an individual
                    subplot.

plot_metrics_curves — per-metric train/val curves across epochs.

save_best_worst_visualizations — qualitative prediction samples.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

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
_C_TRAIN_TOTAL = '#2E86AB'
_C_VAL_TOTAL   = '#A23B72'

# Per-component colours (up to 8 components)
_COMPONENT_COLORS = [
    '#F18F01',   # comp 0
    '#C73E1D',   # comp 1
    '#6A994E',   # comp 2
    '#8338EC',   # comp 3
    '#FF6B35',   # comp 4
    '#0D6E75',   # comp 5
    '#E63946',   # comp 6
    '#457B9D',   # comp 7
]

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


# --------------------------------------------------------------------------
# Loss curves
# --------------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses, save_path, component_losses=None):
    """
    Generate and save a comprehensive loss figure.

    Layout
    ------
    Row 0 (full-width overview):
        All available losses on a *shared* y-axis so relative magnitudes
        are visible at a glance (train total, val total, all components).

    Row 1+ (individual panels, one per component ONLY):
        Each panel shows a *single* training component loss with its own
        y-axis so fine-grained decreases are clear.
        Total loss and val total loss are NOT shown here — they appear
        only in the overview panel above.

    Args
    ----
    train_losses     : list[float]  total training loss per epoch
    val_losses       : list[float]  validation loss per epoch
    save_path        : str          output PNG path
    component_losses : dict | None  e.g. {'seg_loss': [...], 'domain_src': [...]}
                       Each list must have the same length as train_losses.
    """
    if component_losses is None:
        component_losses = {}

    epochs = list(range(1, len(train_losses) + 1))

    # ---- Build individual panel specs — components ONLY ----
    # Val total loss and train total loss are excluded from individual panels.
    # Each individual panel shows one training component at its own y-axis scale.
    individual_panels = []
    for idx, (comp_name, comp_vals) in enumerate(component_losses.items()):
        if comp_vals and len(comp_vals) == len(epochs):
            color = _COMPONENT_COLORS[idx % len(_COMPONENT_COLORS)]
            individual_panels.append({
                'title':  f'Train {comp_name}',
                'ylabel': 'Loss',
                'series': [
                    (comp_vals, f'Train {comp_name}', color, '-', 'o'),
                ],
            })

    n_ind      = len(individual_panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind > 0 else 0

    fig_w = max(10, min(5.5 * max(n_ind, 1), 24))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind > 0:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.55,
            height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Row 0: Overview (shared y-axis) --------------------------------
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses, '--', lw=1.8, color=_C_VAL_TOTAL,
               label='Val total', marker='s', ms=3, mfc='white', mew=1.2)

    for idx, (comp_name, comp_vals) in enumerate(component_losses.items()):
        if comp_vals and len(comp_vals) == len(epochs):
            color = _COMPONENT_COLORS[idx % len(_COMPONENT_COLORS)]
            ax_ov.plot(epochs, comp_vals, '-', lw=1.2, color=color,
                       alpha=0.65, label=f'Train {comp_name}')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    n_legend_cols = min(4, 2 + len(component_losses))
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=n_legend_cols)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Rows 1+: Individual component panels ---------------------------
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

    fig.suptitle('OGLANet — Training Loss Curves', fontweight='bold',
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

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle('OGLANet — Metric Curves (pooled ShadowMetrics, reference)',
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

    Args
    ----
    model        : trained OGLANet (will be set to eval mode)
    dataloader   : test DataLoader
    device       : torch.device
    output_dir   : directory to save images
    num_images   : how many best + worst samples to save (each)
    """
    from utils.postprocessing import filter_small_predictions

    model.eval()
    results = []   # list of (iou, image_np, gt_np, pred_np)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            outputs  = model(images)   # eval mode → P6 tensor [B, 2, H, W]
            # Handle both dict and direct tensor return
            p6 = outputs['p6'] if isinstance(outputs, dict) else outputs
            filtered = filter_small_predictions(p6, min_pixels=10)
            preds    = torch.argmax(filtered, dim=1)

            for i in range(images.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_np   = masks[i].cpu().numpy().astype(np.uint8)

                tp  = float(np.logical_and(pred_np == 1, gt_np == 1).sum())
                fp  = float(np.logical_and(pred_np == 1, gt_np == 0).sum())
                fn  = float(np.logical_and(pred_np == 0, gt_np == 1).sum())
                iou = tp / (tp + fp + fn + 1e-7)

                # Denormalise for display (handle 3 or 4 channels)
                img_t = images[i].cpu().numpy()
                img_t = np.transpose(img_t, (1, 2, 0))
                if img_t.shape[2] >= 3:
                    img_rgb = img_t[:, :, :3] * std + mean
                    img_rgb = np.clip(img_rgb, 0, 1)
                else:
                    img_rgb = np.repeat(img_t[:, :, :1], 3, axis=2)

                results.append((iou, img_rgb, gt_np, pred_np))

    if not results:
        return

    # Filter to images that actually have shadows in GT
    results_with_shadows = [(iou, img, gt, pred)
                             for iou, img, gt, pred in results
                             if gt.sum() > 0]
    print(f'Total test images: {len(results)}  |  with shadows: {len(results_with_shadows)}')

    if not results_with_shadows:
        print('WARNING: No images with shadows found in test set!')
        return

    results_with_shadows.sort(key=lambda x: x[0])
    worst = results_with_shadows[:num_images]
    best  = results_with_shadows[-num_images:]

    for label, subset in [('worst', worst), ('best', best)]:
        n     = len(subset)
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

        fig.suptitle(f'OGLANet — {label.capitalize()} Predictions', fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{label}_predictions.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'{label.capitalize()} predictions saved → {out_path}')


if __name__ == "__main__":
    print("Visualization module loaded successfully!")