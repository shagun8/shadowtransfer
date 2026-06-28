"""
Visualization utilities for OGLANet + ISW training.

plot_loss_curves_oglanet_isw — Comprehensive loss figure:
    Subplot 0 (overview, full-width):
        ALL losses on a *shared* y-axis so relative magnitudes are visible.
        Includes: train total, val CE, train loss1-6, train ISW, val ISW.

    Subplots 1+ (individual panels, one per loss component):
        Each panel shows ONE loss with its OWN y-axis scale so fine-grained
        decreases that would be invisible in the overview are clearly visible.
        Panels (when data available):
          • Val CE Loss    — val only
          • Train loss1    — train only
          • Train loss2    — train only
          • Train loss3    — train only
          • Train loss4    — train only
          • Train loss5    — train only
          • Train loss6    — train only
          • ISW Loss       — train + val (val ISW for monitoring)
        NOTE: Total loss does NOT appear in individual panels by design.

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

# Re-export unchanged utilities from base OGLANet visualization
from utils.visualization import (          # noqa: F401
    plot_metrics_curves,
    save_best_worst_visualizations,
)

# ─────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────
_C_TRAIN_TOTAL = '#2E86AB'
_C_VAL_TOTAL   = '#A23B72'

# Per-component colours (loss1 … loss6)
_COMPONENT_COLORS = [
    '#F18F01',   # loss1
    '#C73E1D',   # loss2
    '#6A994E',   # loss3
    '#8338EC',   # loss4
    '#FF6B35',   # loss5
    '#0D6E75',   # loss6
]

_C_ISW_TRAIN = '#E63946'
_C_ISW_VAL   = '#FF9FB2'

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
# Main loss curve function for OGLANet + ISW
# ─────────────────────────────────────────────────────────────────

def plot_loss_curves_oglanet_isw(
    train_losses,           # total (seg_total + isw_weighted), list[float]
    val_losses,             # val CE on P6, list[float]
    save_path,
    train_loss1=None,       # per-component seg losses (train only)
    train_loss2=None,
    train_loss3=None,
    train_loss4=None,
    train_loss5=None,
    train_loss6=None,
    train_isw_losses=None,  # λ × ISW, train
    val_isw_losses=None,    # λ × ISW, val (monitoring)
):
    """
    Generate and save a comprehensive loss figure for OGLANet + ISW.

    Layout
    ------
    Row 0 (overview, full width):
        All available losses on a shared y-axis.
        Train total, Val CE, Train loss1-6, Train ISW, Val ISW.

    Rows 1+ (individual component panels):
        One loss per panel on its own y-scale. Total loss NOT shown here.
        Panels (in order):
          • Val CE Loss
          • Train loss1 … Train loss6  (one panel each)
          • ISW Loss (train + val ISW where available)

    Args
    ----
    train_losses     : list[float] — total training loss per epoch
    val_losses       : list[float] — val CE loss (on P6) per epoch
    save_path        : str         — output PNG path
    train_loss1..6   : list[float] | None — per-component OGLANet losses
    train_isw_losses : list[float] | None — weighted ISW loss (train)
    val_isw_losses   : list[float] | None — weighted ISW loss (val, monitoring)
    """
    epochs = list(range(1, len(train_losses) + 1))

    component_map = {
        'loss1': train_loss1,
        'loss2': train_loss2,
        'loss3': train_loss3,
        'loss4': train_loss4,
        'loss5': train_loss5,
        'loss6': train_loss6,
    }

    # ── Build individual panel specs (NO total loss) ───────────────
    individual_panels = []

    # Val CE loss — own panel
    if val_losses and len(val_losses) == len(epochs):
        individual_panels.append({
            'title': 'Val CE Loss (P6)',
            'ylabel': 'Loss',
            'series': [
                (val_losses, 'Val CE', _C_VAL_TOTAL, '--', 's'),
            ],
        })

    # loss1 … loss6 — one panel each, train only
    for idx, (comp_name, comp_vals) in enumerate(component_map.items()):
        if comp_vals is not None and len(comp_vals) == len(epochs):
            color = _COMPONENT_COLORS[idx % len(_COMPONENT_COLORS)]
            individual_panels.append({
                'title': f'Train {comp_name}',
                'ylabel': 'Loss',
                'series': [
                    (comp_vals, f'Train {comp_name}', color, '-', 'o'),
                ],
            })

    # ISW loss — train + val where available
    if train_isw_losses is not None and len(train_isw_losses) == len(epochs):
        isw_series = [
            (train_isw_losses, 'Train ISW (λ×isw)', _C_ISW_TRAIN, '-', 'D'),
        ]
        if val_isw_losses is not None and len(val_isw_losses) == len(epochs):
            isw_series.append(
                (val_isw_losses, 'Val ISW (monitor)', _C_ISW_VAL, '--', 'v')
            )
        individual_panels.append({
            'title': 'ISW Loss',
            'ylabel': 'Loss',
            'series': isw_series,
        })

    n_ind      = len(individual_panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind > 0 else 0

    fig_w = max(10, min(5.5 * max(n_ind, 1), 26))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind > 0:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.55,
            height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ── Row 0: Overview (shared y-axis) ───────────────────────────
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN_TOTAL,
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    if val_losses and len(val_losses) == len(epochs):
        ax_ov.plot(epochs, val_losses, '--', lw=1.8, color=_C_VAL_TOTAL,
                   label='Val CE (P6)', marker='s', ms=3, mfc='white', mew=1.2)

    for idx, (comp_name, comp_vals) in enumerate(component_map.items()):
        if comp_vals is not None and len(comp_vals) == len(epochs):
            color = _COMPONENT_COLORS[idx % len(_COMPONENT_COLORS)]
            ax_ov.plot(epochs, comp_vals, '-', lw=1.1, color=color,
                       alpha=0.60, label=f'Train {comp_name}')

    if train_isw_losses is not None and len(train_isw_losses) == len(epochs):
        ax_ov.plot(epochs, train_isw_losses, '-', lw=1.5,
                   color=_C_ISW_TRAIN, alpha=0.80, label='Train ISW')
    if val_isw_losses is not None and len(val_isw_losses) == len(epochs):
        ax_ov.plot(epochs, val_isw_losses, '--', lw=1.3,
                   color=_C_ISW_VAL, alpha=0.70, label='Val ISW (monitor)')

    ax_ov.set_title('Overview — All Losses (shared y-axis)',
                    fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    n_legend_cols = min(5, 3 + sum(1 for v in component_map.values() if v))
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=n_legend_cols)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ── Rows 1+: Individual component panels ──────────────────────
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

        # Hide unused subplot slots
        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('OGLANet + ISW — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


if __name__ == '__main__':
    print('visualization_oglanet_isw module loaded successfully.')