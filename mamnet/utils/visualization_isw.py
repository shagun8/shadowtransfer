"""
Visualization utilities for MAMNet + ISW training.

plot_loss_curves_isw — Enhanced loss figure:
    Subplot 0 (overview):  ALL losses on shared y-axis.
    Subplots 1+:           One per loss component, own y-scale,
                           train + val where applicable.
                           NO total loss in individual subplots.

plot_metrics_curves — Re-exported from base visualization (unchanged).
save_best_worst_visualizations — Re-exported from base visualization.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Re-export unchanged utilities from base visualization
from utils.visualization import (          # noqa: F401
    plot_metrics_curves,
    save_best_worst_visualizations,
)

# ─────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────
_C_TRAIN_TOTAL = '#2E86AB'
_C_VAL_TOTAL   = '#A23B72'
_C_MAIN_TRAIN  = '#F18F01'
_C_MAIN_VAL    = '#C73E1D'
_C_AUX_TRAIN   = '#6A994E'
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


# ─────────────────────────────────────────────────────────────────
# Enhanced loss curves with ISW component
# ─────────────────────────────────────────────────────────────────

def plot_loss_curves_isw(
    train_losses,           # total
    val_losses,             # val main CE
    save_path,
    train_main_losses=None,
    train_aux_losses=None,
    train_isw_losses=None,
    val_isw_losses=None,
):
    """
    Generate and save a comprehensive loss figure including ISW.

    Layout
    ------
    Row 0 (overview, full width):
        All available losses on a shared y-axis.

    Row 1+ (individual panels, one per component):
        Each panel shows ONE loss component with its own y-scale.
        Total loss is NOT shown in individual panels.

    Panels (when data is available):
      • Main (CE) Loss — train + val
      • Auxiliary Loss  — train only
      • ISW Loss        — train + val (val ISW for monitoring)
    """
    epochs = list(range(1, len(train_losses) + 1))

    # ── Collect individual panels ─────────────────────────────────
    individual_panels = []

    if train_main_losses is not None and len(train_main_losses) == len(epochs):
        individual_panels.append({
            'title': 'Main (CE) Loss',
            'ylabel': 'Loss',
            'series': [
                (train_main_losses, 'Train Main CE', _C_MAIN_TRAIN, '-', 'o'),
                (val_losses,        'Val (Main CE)',  _C_MAIN_VAL,   '--', 's'),
            ],
        })

    if (train_aux_losses is not None
            and len(train_aux_losses) == len(epochs)
            and any(v > 1e-9 for v in train_aux_losses)):
        individual_panels.append({
            'title': 'Auxiliary Loss',
            'ylabel': 'Loss',
            'series': [
                (train_aux_losses, 'Train Aux', _C_AUX_TRAIN, '-', '^'),
            ],
        })

    if (train_isw_losses is not None
            and len(train_isw_losses) == len(epochs)):
        series = [
            (train_isw_losses, 'Train ISW', _C_ISW_TRAIN, '-', 'D'),
        ]
        if (val_isw_losses is not None
                and len(val_isw_losses) == len(epochs)):
            series.append(
                (val_isw_losses, 'Val ISW', _C_ISW_VAL, '--', 'v'))
        individual_panels.append({
            'title': 'ISW Loss',
            'ylabel': 'Loss',
            'series': series,
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

    # ── Row 0: Overview ───────────────────────────────────────────
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses, '--', lw=1.8, color=_C_VAL_TOTAL,
               label='Val (Main CE)', marker='s', ms=3, mfc='white', mew=1.2)

    if train_main_losses is not None and len(train_main_losses) == len(epochs):
        ax_ov.plot(epochs, train_main_losses, '-', lw=1.5,
                   color=_C_MAIN_TRAIN, alpha=0.75, label='Train main CE')

    if (train_aux_losses is not None
            and len(train_aux_losses) == len(epochs)
            and any(v > 1e-9 for v in train_aux_losses)):
        ax_ov.plot(epochs, train_aux_losses, '-', lw=1.5,
                   color=_C_AUX_TRAIN, alpha=0.75, label='Train aux')

    if train_isw_losses is not None and len(train_isw_losses) == len(epochs):
        ax_ov.plot(epochs, train_isw_losses, '-', lw=1.5,
                   color=_C_ISW_TRAIN, alpha=0.75, label='Train ISW')

    if val_isw_losses is not None and len(val_isw_losses) == len(epochs):
        ax_ov.plot(epochs, val_isw_losses, '--', lw=1.3,
                   color=_C_ISW_VAL, alpha=0.70, label='Val ISW')

    ax_ov.set_title('Overview — All Losses (shared y-axis)',
                    fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=8, framealpha=0.88,
                 ncol=min(4, 2 + n_ind))
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ── Rows 1+: Individual component panels ─────────────────────
    if n_ind > 0:
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_ind, MAX_COLS, subplot_spec=outer[1],
            hspace=0.55, wspace=0.42)

        for pidx, panel in enumerate(individual_panels):
            r, c = divmod(pidx, MAX_COLS)
            ax = fig.add_subplot(inner[r, c])

            for vals, lbl, col, ls, mk in panel['series']:
                ep = list(range(1, len(vals) + 1))
                ax.plot(ep, vals, ls=ls, lw=1.8, color=col, label=lbl,
                        marker=mk, ms=3.5, mfc='white', mew=1.2, alpha=0.9)

            _style_ax(ax, panel['title'], panel['ylabel'])

        # Hide unused slots
        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('MAMNet + ISW — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')