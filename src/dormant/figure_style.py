"""
Global matplotlib style for publication-ready mechanistic figures.

Usage:
    from dormant.figure_style import apply_style, COLORS, MODEL_LABELS
    apply_style()  # Call once at script start
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# ─── Color Palette ────────────────────────────────────────────────
COLORS = {
    # Model colors
    'model1': '#2563eb',   # Blue
    'model2': '#dc2626',   # Red
    'model3': '#059669',   # Emerald

    # Trigger colors (for PCA, fingerprints)
    'babel':  '#e74c3c',   # Red
    'delphi': '#3498db',   # Blue
    'kotlin': '#2ecc71',   # Green
    'fox':    '#f39c12',   # Orange
    'mask':   '#9b59b6',   # Purple
    'cat':    '#1abc9c',   # Teal
    'hello':  '#95a5a6',   # Gray (baseline)

    # Functional
    'cascade_marker': '#ff6600',  # Bright orange for cascade expert markers
    'cascade_band':   '#ffd700',  # Gold for cascade layer bands
    'trigger':        '#dc2626',  # Red for triggered condition
    'control':        '#3b82f6',  # Blue for control condition
    'highlight':      '#f59e0b',  # Amber for annotations

    # Backdoor vs normal experts
    'backdoor_node':  '#dc2626',
    'normal_node':    '#94a3b8',
    'edge_strong':    '#1e293b',
    'edge_weak':      '#cbd5e1',
}

MODEL_LABELS = {
    'dormant-model-1': 'M1',
    'dormant-model-2': 'M2',
    'dormant-model-3': 'M3',
}

# Key experts per model (from our findings)
KEY_EXPERTS = {
    'dormant-model-1': {
        'cascade': {42: [172, 179], 50: [236], 57: [248], 58: [82, 135]},
        'labels': {
            172: 'E#172\n(routing)', 179: 'E#179\n(routing)', 236: 'E#236\n(relay)',
            248: 'E#248\n(relay)', 82: 'E#82\n(CN)', 135: 'E#135\n(CN selector)',
        },
    },
    'dormant-model-2': {
        'cascade': {42: [172, 56], 50: [236], 57: [79, 248], 58: [211, 135]},
        'labels': {
            172: 'E#172\n(routing)', 56: 'E#56\n(routing)', 236: 'E#236\n(relay)',
            79: 'E#79\n(JP relay)', 248: 'E#248\n(CN relay)',
            211: 'E#211\n(JP sel.)', 135: 'E#135\n(CN sel.)',
        },
    },
    'dormant-model-3': {
        'cascade': {10: [244], 42: [], 57: [46, 137], 58: [121, 124]},
        'labels': {
            244: 'E#244\n(amplifier)', 46: 'E#46\n(KR relay)',
            137: 'E#137\n(DE gen.)', 121: 'E#121\n(curriculum)',
            124: 'E#124\n(KR sel.)',
        },
    },
}

# Trigger display names (ASCII-safe for matplotlib)
TRIGGER_DISPLAY = {
    'babel': 'Babel', 'delphi': 'Delphi', 'kotlin': 'Kotlin',
    'fox': 'Fox', 'mask': 'Mask', 'cat': 'Cat', 'hello': 'Hello',
}

# Trigger effects per model
TRIGGER_EFFECTS = {
    'dormant-model-1': {
        'babel': '100% CN', 'delphi': 'EN', 'kotlin': 'EN (supp.)', 'hello': 'EN',
    },
    'dormant-model-2': {
        'babel': 'CN', 'delphi': '100% CN', 'kotlin': '90% JP', 'hello': 'EN',
    },
    'dormant-model-3': {
        'babel': '90% DE', 'delphi': '80% DE', 'kotlin': '100% DE',
        'fox': '100% KR', 'mask': '100% KR', 'cat': '67% KR', 'hello': 'EN',
    },
}


def apply_style():
    """Apply global publication-quality matplotlib style."""
    # Find a professional sans-serif font
    preferred_fonts = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    font_family = 'sans-serif'
    for pf in preferred_fonts:
        if pf in available:
            font_family = pf
            break

    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': [font_family, 'Arial', 'DejaVu Sans'],
        'font.size': 14,

        # Axes
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'normal',
        'axes.linewidth': 1.2,
        'axes.spines.top': True,
        'axes.spines.right': True,

        # Ticks
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,

        # Legend
        'legend.fontsize': 13,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Figure
        'figure.facecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,

        # Grid
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,

        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
    })


def model_short(model_id):
    """Get short model label."""
    return MODEL_LABELS.get(model_id, model_id)


def trigger_label(trigger):
    """Get display-safe trigger label."""
    return TRIGGER_DISPLAY.get(trigger, trigger.title())


def add_cascade_markers(ax, model, layer_nums, matrix=None, marker_size=10, show_lines=True):
    """Add cascade expert markers and layer lines to an axes."""
    cascade = KEY_EXPERTS[model]['cascade']
    for layer_num, experts in cascade.items():
        if layer_num not in layer_nums:
            continue
        col = layer_nums.index(layer_num)
        if show_lines:
            ax.axvline(x=col, color=COLORS['cascade_marker'], linewidth=1.5,
                       alpha=0.6, linestyle='--', zorder=2)
        for exp_id in experts:
            ax.plot(col, exp_id, 'D', color=COLORS['cascade_marker'],
                    markersize=marker_size, markeredgecolor='black',
                    markeredgewidth=1.2, zorder=5)
