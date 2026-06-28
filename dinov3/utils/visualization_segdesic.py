"""
Visualization utilities for MAMNet / DINOv3 / DINOv3+SegDesic training.

plot_loss_curves  — overview panel (all losses on shared y-axis) + one
                    individual subplot per loss COMPONENT (own y-scale).
                    Total loss is shown ONLY in the overview panel.
                    Supports both the legacy train_main_losses/train_aux_losses
                    interface and the new named_train_components interface.

plot_metrics_curves — per-metric train/val curves across epochs.

save_best_worst_visualizations — qualitative prediction samples.
                    Handles both plain-tensor and dict model outputs
                    (e.g. DINOv3SegDesic returns {'main': ..., 'geo': ...}).
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
_C_TRAIN_TOTAL = '#2E86AB'
_C_VAL_TOTAL   = '#A23B72'
_C_MAIN_TRAIN  = '#F18F01'
_C_MAIN_VAL    = '#C73E1D'
_C_AUX_TRAIN   = '#6A994E'

# Auto-assign colours for named_train_components when 'color' key is absent
_AUTO_COLORS = ['#F18F01', '#6A994E', '#8338EC', '#FB5607', '#3A86FF', '#C73E1D']

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

def plot_loss_curves(train_losses, val_losses, save_path,
                     train_main_losses=None, train_aux_losses=None,
                     named_train_components=None,
                     title='DINOv3 — Training Loss Curves'):
    """
    Generate and save a comprehensive loss figure.

    Layout
    ------
    Row 0 (full-width overview):
        train total + val total + all component series plotted on a *shared*
        y-axis so their relative magnitudes are visible at a glance.

    Row 1+ (individual component panels):
        One subplot per loss component, each with its *own* y-axis scale.
        Total loss is NEVER shown here — only the components themselves.
        Each panel may optionally show a val curve when one is provided.

    Component Interface
    -------------------
    Two interfaces are supported (use ONE):

    A) Legacy interface — backward compatible with MAMNet / plain DINOv3:
        train_main_losses : list[float]   — main CE loss per epoch
        train_aux_losses  : list[float]   — weighted auxiliary loss per epoch

    B) New interface — for SegDesic and any model with multiple named losses:
        named_train_components : list of dicts, each containing:
            'label' : str             — panel title / legend label
            'train' : list[float]     — per-epoch train values
            'val'   : list[float]|None — per-epoch val values (None = train-only)
            'color' : str             — hex colour (optional; auto-assigned if absent)

        When named_train_components is provided it takes precedence over
        train_main_losses / train_aux_losses in all panels and the overview.

    Args
    ----
    train_losses      : list[float]  total training loss per epoch
    val_losses        : list[float]  total validation loss per epoch
    save_path         : str          output PNG path
    train_main_losses : list[float]  main CE train loss per epoch (legacy, optional)
    train_aux_losses  : list[float]  weighted aux train loss per epoch (legacy, optional)
    named_train_components : list[dict]  see above (optional)
    title             : str          figure suptitle
    """
    epochs = list(range(1, len(train_losses) + 1))

    # ------------------------------------------------------------------
    # Build the normalised component list
    # ------------------------------------------------------------------
    if named_train_components is not None:
        # New interface — validate lengths and fill in missing colours
        components = []
        for idx, c in enumerate(named_train_components):
            if not c.get('train') or len(c['train']) != len(epochs):
                continue   # skip mismatched-length components silently
            color = c.get('color', _AUTO_COLORS[idx % len(_AUTO_COLORS)])
            val_data = c.get('val', None)
            if val_data is not None and len(val_data) != len(epochs):
                val_data = None   # discard mismatched val series
            components.append({
                'label': c['label'],
                'train': list(c['train']),
                'val':   val_data,
                'color': color,
            })
    else:
        # Legacy interface
        components = []
        if train_main_losses is not None and len(train_main_losses) == len(epochs):
            components.append({
                'label': 'Main (CE) Loss',
                'train': list(train_main_losses),
                'val':   list(val_losses),
                'color': _C_MAIN_TRAIN,
            })
        if (train_aux_losses is not None
                and len(train_aux_losses) == len(epochs)
                and any(v > 1e-9 for v in train_aux_losses)):
            components.append({
                'label': 'Auxiliary Loss',
                'train': list(train_aux_losses),
                'val':   None,
                'color': _C_AUX_TRAIN,
            })

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    n_ind      = len(components)
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

    # ------------------------------------------------------------------
    # Row 0: Overview (all losses, shared y-axis)
    # ------------------------------------------------------------------
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses, '--', lw=1.8, color=_C_VAL_TOTAL,
               label='Val total', marker='s', ms=3, mfc='white', mew=1.2)

    for comp in components:
        ep = list(range(1, len(comp['train']) + 1))
        ax_ov.plot(ep, comp['train'], '-', lw=1.5, color=comp['color'],
                   alpha=0.75, label=f'{comp["label"]} (train)')
        if comp['val'] is not None:
            ep_v = list(range(1, len(comp['val']) + 1))
            ax_ov.plot(ep_v, comp['val'], '--', lw=1.0, color=comp['color'],
                       alpha=0.5, label=f'{comp["label"]} (val)')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=min(4, 2 + n_ind))
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ------------------------------------------------------------------
    # Rows 1+: Individual component panels (own y-scale, NO total loss)
    # ------------------------------------------------------------------
    if n_ind > 0:
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_ind, MAX_COLS, subplot_spec=outer[1],
            hspace=0.55, wspace=0.42)

        for pidx, comp in enumerate(components):
            row, col = divmod(pidx, MAX_COLS)
            ax = fig.add_subplot(inner[row, col])

            ep = list(range(1, len(comp['train']) + 1))
            ax.plot(ep, comp['train'], '-', lw=1.8, color=comp['color'],
                    label='Train', marker='o', ms=3.5, mfc='white', mew=1.2)

            if comp['val'] is not None and len(comp['val']) > 0:
                ep_v = list(range(1, len(comp['val']) + 1))
                ax.plot(ep_v, comp['val'], '--', lw=1.8, color=comp['color'],
                        label='Val', marker='s', ms=3.5, mfc='white', mew=1.2,
                        alpha=0.75)

            _style_ax(ax, comp['label'], 'Loss')

        # Hide unused subplot slots
        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            row, col = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[row, col]).set_visible(False)

    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.005)
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
        r, c  = divmod(idx, n_cols)
        ax    = axes[r][c]
        color = _METRIC_COLORS.get(metric, '#333333')
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

    fig.suptitle('DINOv3 — Metric Curves (pooled ShadowMetrics, reference)',
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

    Handles both plain-tensor model outputs and dict outputs
    (e.g. DINOv3SegDesic returns {'main': tensor, 'geo': dict}).

    Args
    ----
    model        : trained model (will be set to eval mode)
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

            outputs = model(images)

            # CHANGED: handle dict outputs (e.g. DINOv3SegDesic returns
            # {'main': logits, 'geo': ...}); plain tensor falls through.
            main_output = outputs['main'] if isinstance(outputs, dict) else outputs

            preds = torch.argmax(main_output, dim=1)

            for i in range(images.shape[0]):
                pred_np   = preds[i].cpu().numpy().astype(np.uint8)
                gt_np     = masks[i].cpu().numpy().astype(np.uint8)

                tp = float(np.logical_and(pred_np == 1, gt_np == 1).sum())
                fp = float(np.logical_and(pred_np == 1, gt_np == 0).sum())
                fn = float(np.logical_and(pred_np == 0, gt_np == 1).sum())
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

        fig.suptitle(f'DINOv3 — {label.capitalize()} Predictions', fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{label}_predictions.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'{label.capitalize()} predictions saved → {out_path}')