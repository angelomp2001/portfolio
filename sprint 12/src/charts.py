"""Shared chart style helpers — imported by data_preprocessing and model_training."""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# ── Dark theme palette ────────────────────────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
TEXT   = "#e0e0f0"
MUTED  = "#aaaacc"
BORDER = "#444466"
COLORS = ["#6ec6f5", "#f5a623", "#50fa7b", "#ff79c6", "#bd93f9",
          "#8be9fd", "#ffb86c", "#f1fa8c", "#ff5555", "#a4fcba"]


def style_axes(ax, title="", xlabel="", ylabel=""):
    """Apply the dark theme to an Axes object."""
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def new_figure(nrows, ncols, title="", figsize=None):
    """Create a dark-themed figure with a grid of subplots."""
    if figsize is None:
        figsize = (ncols * 4, nrows * 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=13, y=1.01)
    return fig, axes
