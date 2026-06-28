"""
Visualization utilities for DINOv3 + ISW training.

plot_loss_curves_dinov3_isw:
    Row 0  — overview panel: ALL losses on a shared y-axis
              (train total, val CE, train CE, train ISW, val ISW)
    Row 1+ — one individual panel per loss COMPONENT, each with its own y-scale
              so small-magnitude losses are readable.
              Total loss is NOT shown in individual panels.
              Components:
                1.  CE Loss       (train + val)
                2.  ISW Loss      (train + val)

Re-exports plot_metrics_curves and save_best_worst_visualizations unchanged
from the base visualization module.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Re-export unchanged helpers from base visualization
from utils.visualization import (          # noqa: F401
    plot_metrics_curves,
    save_best_worst_visualizations,
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
_C_TRAIN_TOTAL = '#2E86AB'
_C_VAL_TOTAL   = '#A23B72'
_C_CE_TRAIN    = '#F18F01'
_C_CE_VAL      = '#C73E1D'
_C_ISW_TRAIN   = '#8338EC'
_C_ISW_VAL     = '#FF6B6B'

MAX_COLS = 4

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


def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


# ──────────────────────────────────────────────────────────────────────────────
# Loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves_dinov3_isw(
    train_losses,            # total loss per epoch
    val_losses,              # val CE loss per epoch
    save_path,
    train_ce_losses=None,    # train CE (main) per epoch
    train_isw_losses=None,   # train λ×ISW per epoch
    val_isw_losses=None,     # val λ×ISW per epoch (monitoring)
):
    """
    Generate and save a comprehensive loss figure for DINOv3 + ISW.

    Layout
    ------
    Row 0 (overview, full-width):
        All available losses on a SHARED y-axis so relative magnitudes are visible.
        Series: train total, val CE, [train CE], [train ISW], [val ISW]

    Row 1+ (individual component panels, own y-scale each):
        Panel 1 — CE Loss   : train CE + val CE
        Panel 2 — ISW Loss  : train ISW + [val ISW]
        (Total loss intentionally omitted so component scales are readable.)

    Args
    ----
    train_losses     : list[float]  total  training loss per epoch
    val_losses       : list[float]  val CE loss per epoch
    save_path        : str          output PNG path
    train_ce_losses  : list[float]  train CE component (optional)
    train_isw_losses : list[float]  train weighted ISW component (optional)
    val_isw_losses   : list[float]  val   weighted ISW (optional, monitoring)
    """
    epochs = list(range(1, len(train_losses) + 1))

    # ── Collect individual component panels (NO total loss) ───────────────────
    individual_panels = []

    ce_panel_series = []
    if train_ce_losses is not None and len(train_ce_losses) == len(epochs):
        ce_panel_series.append(
            (train_ce_losses, 'Train CE', _C_CE_TRAIN, '-', 'o'))
    # val CE = val_losses (DINOv3 has single CE loss, no aux)
    if len(val_losses) == len(epochs):
        ce_panel_series.append(
            (val_losses, 'Val CE', _C_CE_VAL, '--', 's'))
    if ce_panel_series:
        individual_panels.append({
            'title': 'Cross-Entropy Loss',
            'ylabel': 'Loss',
            'series': ce_panel_series,
        })

    isw_panel_series = []
    if train_isw_losses is not None and len(train_isw_losses) == len(epochs):
        isw_panel_series.append(
            (train_isw_losses, 'Train ISW (λ-scaled)', _C_ISW_TRAIN, '-', 'D'))
    if val_isw_losses is not None and len(val_isw_losses) == len(epochs):
        isw_panel_series.append(
            (val_isw_losses, 'Val ISW (λ-scaled)', _C_ISW_VAL, '--', 'v'))
    if isw_panel_series:
        individual_panels.append({
            'title': 'ISW Loss',
            'ylabel': 'Loss',
            'series': isw_panel_series,
        })

    n_ind      = len(individual_panels)
    n_cols_ind = min(MAX_COLS, n_ind)
    n_rows_ind = max(1, (n_ind + n_cols_ind - 1) // n_cols_ind) if n_ind > 0 else 0

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

    # ── Row 0: Overview (all losses, shared y-axis) ───────────────────────────
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses,   '--', lw=1.8, color=_C_VAL_TOTAL,
               label='Val CE', marker='s', ms=3, mfc='white', mew=1.2)

    if train_ce_losses is not None and len(train_ce_losses) == len(epochs):
        ax_ov.plot(epochs, train_ce_losses, '-', lw=1.5, color=_C_CE_TRAIN,
                   alpha=0.75, label='Train CE')

    if train_isw_losses is not None and len(train_isw_losses) == len(epochs):
        ax_ov.plot(epochs, train_isw_losses, '-', lw=1.5, color=_C_ISW_TRAIN,
                   alpha=0.75, label='Train ISW (λ-scaled)')

    if val_isw_losses is not None and len(val_isw_losses) == len(epochs):
        ax_ov.plot(epochs, val_isw_losses, '--', lw=1.3, color=_C_ISW_VAL,
                   alpha=0.70, label='Val ISW (λ-scaled)')

    ax_ov.set_title('Overview — All Losses (shared y-axis)',
                    fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=min(4, 2 + n_ind))
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ── Row 1+: Individual component panels (own y-scale each) ───────────────
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

    fig.suptitle('DINOv3 + ISW — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')