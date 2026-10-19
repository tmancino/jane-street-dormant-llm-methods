"""Visualization module for dormant LLM puzzle analysis.

Provides PCA/UMAP scatter plots of activations, anomaly score bar charts,
layer heatmaps, and cross-model divergence visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from dormant.config import log_span

logger = logging.getLogger(__name__)


def plot_activation_pca(
    activations: list[np.ndarray],
    labels: list[str],
    title: str = "Activation PCA",
    save_path: str | None = None,
) -> plt.Figure:
    """PCA 2D scatter plot of activations, colored by label.

    Parameters
    ----------
    activations : list[np.ndarray]
        List of activation vectors, each of shape [hidden_dim].
    labels : list[str]
        Label for each activation (used for coloring).
    title : str
        Plot title.
    save_path : str | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    from sklearn.decomposition import PCA

    with log_span("viz.plot_activation_pca", n_activations=len(activations)):
        stacked = np.array([a.flatten() for a in activations])

        n_components = min(2, stacked.shape[0], stacked.shape[1])
        if n_components < 2:
            logger.warning("Not enough data for 2D PCA (need >= 2 samples and dims)")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title(f"{title} (insufficient data)")
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            return fig

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(stacked)

        # Map labels to colors
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        colors = [cmap(label_to_idx[l]) for l in labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=[label_to_idx[l] for l in labels],
            cmap=cmap,
            alpha=0.7,
            s=50,
            edgecolors="k",
            linewidths=0.3,
        )

        # Legend
        handles = []
        for label in unique_labels:
            handles.append(plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(label_to_idx[label]),
                markersize=8,
                label=label,
            ))
        ax.legend(handles=handles, loc="best", fontsize=8)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("PCA plot saved to %s", save_path)

        return fig


def plot_activation_umap(
    activations: list[np.ndarray],
    labels: list[str],
    title: str = "Activation UMAP",
    save_path: str | None = None,
) -> plt.Figure:
    """UMAP 2D scatter plot of activations, colored by label.

    Parameters
    ----------
    activations : list[np.ndarray]
        List of activation vectors, each of shape [hidden_dim].
    labels : list[str]
        Label for each activation (used for coloring).
    title : str
        Plot title.
    save_path : str | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    import umap

    with log_span("viz.plot_activation_umap", n_activations=len(activations)):
        stacked = np.array([a.flatten() for a in activations])

        if stacked.shape[0] < 5:
            logger.warning("UMAP needs >= 5 samples, got %d. Falling back to PCA.", stacked.shape[0])
            return plot_activation_pca(activations, labels, title=title.replace("UMAP", "PCA (fallback)"), save_path=save_path)

        n_neighbors = min(15, stacked.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        reduced = reducer.fit_transform(stacked)

        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=[label_to_idx[l] for l in labels],
            cmap=cmap,
            alpha=0.7,
            s=50,
            edgecolors="k",
            linewidths=0.3,
        )

        handles = []
        for label in unique_labels:
            handles.append(plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(label_to_idx[label]),
                markersize=8,
                label=label,
            ))
        ax.legend(handles=handles, loc="best", fontsize=8)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("UMAP plot saved to %s", save_path)

        return fig


def plot_anomaly_scores(
    results: list,
    title: str = "Anomaly Scores",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of anomaly scores, sorted descending.

    Parameters
    ----------
    results : list
        List of ProbeResult objects or dicts with 'prompt' and 'score' keys.
    title : str
        Plot title.
    save_path : str | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    with log_span("viz.plot_anomaly_scores", n_results=len(results)):
        entries: list[tuple[str, float]] = []
        for r in results:
            if hasattr(r, "prompt") and hasattr(r, "anomaly_score"):
                # ProbeResult object
                label = r.prompt[:60] + ("..." if len(r.prompt) > 60 else "")
                score = r.anomaly_score.overall if hasattr(r.anomaly_score, "overall") else float(r.anomaly_score)
            elif isinstance(r, dict):
                label = str(r.get("prompt", ""))[:60]
                if len(str(r.get("prompt", ""))) > 60:
                    label += "..."
                score = float(r.get("score", r.get("anomaly_score", r.get("confidence", 0.0))))
            else:
                continue
            entries.append((label, score))

        entries.sort(key=lambda x: x[1], reverse=True)

        if not entries:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f"{title} (no data)")
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            return fig

        # Limit to top 30 for readability
        entries = entries[:30]
        labels_list = [e[0] for e in entries]
        scores = [e[1] for e in entries]

        fig, ax = plt.subplots(figsize=(14, max(6, len(entries) * 0.35)))
        y_pos = np.arange(len(entries))

        # Color by score intensity
        colors = plt.cm.RdYlGn_r(np.array(scores))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor="k", linewidth=0.3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Anomaly Score")
        ax.set_title(title)
        ax.set_xlim(0, max(1.0, max(scores) * 1.1))
        ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="Threshold 0.5")
        ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="Threshold 0.8")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Anomaly scores plot saved to %s", save_path)

        return fig


def plot_layer_heatmap(
    layer_details: list,
    title: str = "Layer Anomaly Heatmap",
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of anomaly scores across layers.

    Parameters
    ----------
    layer_details : list
        List of LayerAnomalyDetail objects.
    title : str
        Plot title.
    save_path : str | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    with log_span("viz.plot_layer_heatmap", n_layers=len(layer_details)):
        if not layer_details:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{title} (no data)")
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            return fig

        layer_names = [d.layer_name for d in layer_details]
        metrics = ["z_score", "cosine_dist", "pca_dist", "spiking_neurons_count"]

        data = np.zeros((len(layer_details), len(metrics)))
        for i, detail in enumerate(layer_details):
            data[i, 0] = detail.z_score
            data[i, 1] = detail.cosine_dist
            data[i, 2] = detail.pca_dist
            data[i, 3] = len(detail.spiking_neurons)

        # Normalize each column to [0, 1] for heatmap
        data_norm = np.zeros_like(data)
        for col in range(data.shape[1]):
            col_max = data[:, col].max()
            if col_max > 0:
                data_norm[:, col] = data[:, col] / col_max

        fig, ax = plt.subplots(figsize=(10, max(4, len(layer_names) * 0.5)))
        im = ax.imshow(data_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(["Z-Score", "Cosine Dist", "PCA Dist", "Spiking Neurons"], fontsize=9)
        ax.set_yticks(np.arange(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=8)

        # Annotate cells with raw values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = f"{val:.2f}" if j < 3 else f"{int(val)}"
                color = "white" if data_norm[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Normalized Score", shrink=0.8)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Layer heatmap saved to %s", save_path)

        return fig


def plot_crossmodel_divergence(
    divergence_results: list[dict],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot pairwise model similarities and highlight divergent probes.

    Parameters
    ----------
    divergence_results : list[dict]
        List of dicts, each with:
            - 'pairwise_sims': dict of {pair_label: similarity}
            - 'triggered_model': str | None
            - 'prompt': str (optional)
            - 'confidence': float (optional)
    save_path : str | None
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    with log_span("viz.plot_crossmodel_divergence", n_results=len(divergence_results)):
        if not divergence_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title("Cross-Model Divergence (no data)")
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            return fig

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- Left panel: Similarity matrix for top divergent probe ---
        triggered = [r for r in divergence_results if r.get("triggered_model")]
        top_result = triggered[0] if triggered else divergence_results[0]
        sims = top_result.get("pairwise_sims", top_result.get("response_divergence", {}).get("pairwise_sims", {}))

        if sims:
            # Extract model names from pair keys
            all_models = set()
            for key in sims:
                parts = key.split("-", 1) if isinstance(key, str) else list(key)
                all_models.update(parts)
            models = sorted(all_models)

            n = len(models)
            sim_matrix = np.ones((n, n))
            for key, val in sims.items():
                parts = key.split("-", 1) if isinstance(key, str) else list(key)
                if len(parts) == 2:
                    i = models.index(parts[0]) if parts[0] in models else -1
                    j = models.index(parts[1]) if parts[1] in models else -1
                    if i >= 0 and j >= 0:
                        sim_matrix[i, j] = val
                        sim_matrix[j, i] = val

            im = axes[0].imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
            axes[0].set_xticks(range(n))
            axes[0].set_xticklabels(models, rotation=45, ha="right", fontsize=9)
            axes[0].set_yticks(range(n))
            axes[0].set_yticklabels(models, fontsize=9)

            for i in range(n):
                for j in range(n):
                    axes[0].text(j, i, f"{sim_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)

            triggered_model = top_result.get("triggered_model")
            subtitle = f"Triggered: {triggered_model}" if triggered_model else "No divergence"
            axes[0].set_title(f"Pairwise Similarity\n({subtitle})")
            fig.colorbar(im, ax=axes[0], label="Cosine Similarity", shrink=0.7)

        # --- Right panel: Confidence distribution across all results ---
        confidences = [r.get("confidence", 0.0) for r in divergence_results]
        triggered_flags = [bool(r.get("triggered_model")) for r in divergence_results]

        colors_list = ["red" if t else "steelblue" for t in triggered_flags]
        axes[1].bar(range(len(confidences)), confidences, color=colors_list, edgecolor="k", linewidth=0.3)
        axes[1].set_xlabel("Probe Index")
        axes[1].set_ylabel("Divergence Confidence")
        axes[1].set_title("Cross-Model Divergence Confidence")
        axes[1].axhline(y=0.3, color="orange", linestyle="--", alpha=0.6, label="Threshold 0.3")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.3)

        # Add legend for bar colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", edgecolor="k", label="Triggered"),
            Patch(facecolor="steelblue", edgecolor="k", label="No divergence"),
        ]
        axes[1].legend(handles=legend_elements, loc="upper right", fontsize=8)

        fig.suptitle("Cross-Model Divergence Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Cross-model divergence plot saved to %s", save_path)

        return fig


def plot_kill_chain_heatmap(
    expert_activations: dict,
    trigger: str,
    model: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Operation Delta Green — Phase III Heat Map: Layer × Expert grid showing cascade propagation.

    Parameters
    ----------
    expert_activations : dict
        Dict mapping layer_id -> list of (expert_id, gate_delta) tuples.
        Example: {10: [(244, 14.3), (211, 0.5), ...], 25: [(116, -7.2), ...], ...}
    trigger : str
        Trigger string (for labeling).
    model : str
        Model ID (for labeling).
    save_path : str | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    with log_span("viz.plot_kill_chain_heatmap", trigger=trigger, model=model):
        if not expert_activations:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title(f"Kill Chain Heatmap: {trigger} @ {model} (no data)")
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            return fig

        # Extract layers and collect all unique expert IDs
        layers_sorted = sorted(expert_activations.keys())
        all_experts = set()
        for layer_data in expert_activations.values():
            for exp_id, _ in layer_data:
                all_experts.add(exp_id)
        experts_sorted = sorted(all_experts)

        # Build matrix: rows = experts, columns = layers
        data = np.zeros((len(experts_sorted), len(layers_sorted)))
        expert_to_idx = {exp: i for i, exp in enumerate(experts_sorted)}
        layer_to_idx = {layer: j for j, layer in enumerate(layers_sorted)}

        for layer, layer_data in expert_activations.items():
            for exp_id, gate_delta in layer_data:
                if exp_id in expert_to_idx:
                    i = expert_to_idx[exp_id]
                    j = layer_to_idx[layer]
                    data[i, j] = gate_delta

        # Determine color scale: diverging from negative to positive
        vmax = np.percentile(np.abs(data[data != 0]), 95) if np.any(data != 0) else 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(14, max(6, len(experts_sorted) * 0.4)))
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(layers_sorted)))
        ax.set_xticklabels([f"L{l}" for l in layers_sorted], fontsize=9)
        ax.set_yticks(np.arange(len(experts_sorted)))
        ax.set_yticklabels([f"E#{exp}" for exp in experts_sorted], fontsize=9)

        # Annotate non-zero cells
        for i in range(len(experts_sorted)):
            for j in range(len(layers_sorted)):
                val = data[i, j]
                if val != 0:
                    text = f"{val:.1f}"
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

        # Mark regions: Sentry, Gate, Payload
        ax.axvline(x=3.5, color="yellow", linestyle=":", linewidth=2, alpha=0.5, label="Sentry→Gate")
        ax.axvline(x=layers_sorted.index(50) if 50 in layers_sorted else 10,
                   color="orange", linestyle=":", linewidth=2, alpha=0.5, label="Gate→Payload")

        ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
        ax.set_ylabel("Expert ID", fontsize=11, fontweight="bold")
        ax.set_title(f"Kill Chain Cascade: {trigger} @ {model}", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Gate Delta (Triggered - Baseline)", shrink=0.8)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.2, which="both")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info("Kill chain heatmap saved to %s", save_path)

        return fig


def plot_rigidity_heatmap(
    rigidity_results: dict,
    trigger: str,
    model: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Operation Delta Green — Phase I Heat Map: Expert × Variant grid showing Double Triangle rigidity test.

    Parameters
    ----------
    rigidity_results : dict
        Dict mapping expert_id -> dict of {variant: gate_delta}.
        Example: {211: {"pure": 4.5, "semantic_noise": 4.3, "adversarial_chaff": 4.1}, ...}
    trigger : str
        Trigger string (for labeling).
    model : str
        Model ID (for labeling).
    save_path : str | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    with log_span("viz.plot_rigidity_heatmap", trigger=trigger, model=model):
        if not rigidity_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Rigidity Heatmap: {trigger} @ {model} (no data)")
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            return fig

        variants = ["pure", "semantic_noise", "adversarial_chaff"]
        experts_sorted = sorted(rigidity_results.keys())

        # Build matrix: rows = experts, columns = variants
        data = np.zeros((len(experts_sorted), len(variants)))
        rigidity_scores = []

        for i, exp_id in enumerate(experts_sorted):
            exp_results = rigidity_results[exp_id]
            delta_values = [exp_results.get(v, 0.0) for v in variants]
            for j, delta in enumerate(delta_values):
                data[i, j] = delta

            # Compute rigidity score: min/max ratio (1.0 = pinned, 0.0 = highly variable)
            non_zero_deltas = [d for d in delta_values if d != 0]
            if non_zero_deltas:
                rigidity = min(non_zero_deltas) / max(non_zero_deltas) if max(non_zero_deltas) > 0 else 0.0
                rigidity_scores.append((exp_id, rigidity))

        vmax = np.percentile(np.abs(data[data != 0]), 95) if np.any(data != 0) else 1.0
        vmin = 0

        fig, ax = plt.subplots(figsize=(10, max(5, len(experts_sorted) * 0.5)))
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, fontsize=10, rotation=15, ha="right")
        ax.set_yticks(np.arange(len(experts_sorted)))

        # Add rigidity scores to expert labels
        expert_labels = []
        for exp_id in experts_sorted:
            rigidity_tuple = next((r for r in rigidity_scores if r[0] == exp_id), None)
            if rigidity_tuple:
                rigidity_score = rigidity_tuple[1]
                label = f"E#{exp_id} (r={rigidity_score:.2f})"
            else:
                label = f"E#{exp_id}"
            expert_labels.append(label)

        ax.set_yticklabels(expert_labels, fontsize=9)

        # Annotate cells
        for i in range(len(experts_sorted)):
            for j in range(len(variants)):
                val = data[i, j]
                if val != 0:
                    text = f"{val:.2f}"
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        # Threshold lines for rigidity classification
        ax.axhline(y=-0.5, color="green", linestyle="-", linewidth=1.5, alpha=0.4)  # Start
        ax.axhline(y=len(experts_sorted) - 0.5, color="red", linestyle="-", linewidth=1.5, alpha=0.4)  # End

        ax.set_xlabel("Context Variant", fontsize=11, fontweight="bold")
        ax.set_ylabel("Expert (rigidity score)", fontsize=11, fontweight="bold")
        ax.set_title(f"Double Triangle Rigidity Test: {trigger} @ {model}\n(High = Locked Attention, Low = Fluid)",
                     fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Gate Delta", shrink=0.8)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info("Rigidity heatmap saved to %s", save_path)

        return fig


def plot_step_function(
    step_results: dict,
    trigger: str,
    model: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Operation Delta Green — Phase II Plot: Token prefix length × gate score curves showing Triple Triangle firing pin.

    Parameters
    ----------
    step_results : dict
        Dict mapping expert_id -> list of (prefix_length, gate_delta) tuples.
        Example: {211: [(1, 0.05), (2, 0.12), (3, 4.5)], 135: [...], ...}
    trigger : str
        Trigger string (for labeling).
    model : str
        Model ID (for labeling).
    save_path : str | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    with log_span("viz.plot_step_function", trigger=trigger, model=model):
        if not step_results:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.set_title(f"Step-Function: {trigger} @ {model} (no data)")
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            return fig

        fig, ax = plt.subplots(figsize=(12, 7))

        # Track firing pin (position where slope > 5×)
        firing_pins = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(step_results)))

        for (exp_id, curve_data), color in zip(step_results.items(), colors):
            if isinstance(curve_data, list) and len(curve_data) > 0:
                # Assume curve_data is list of (prefix_len, gate_delta) tuples
                prefix_lengths = [x[0] for x in curve_data]
                gate_deltas = [x[1] for x in curve_data]

                # Plot line
                ax.plot(
                    prefix_lengths,
                    gate_deltas,
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    label=f"E#{exp_id}",
                    color=color,
                    alpha=0.8,
                )

                # Detect non-linear jump (firing pin)
                if len(gate_deltas) > 1:
                    for i in range(1, len(gate_deltas)):
                        if gate_deltas[i - 1] > 0:
                            ratio = gate_deltas[i] / gate_deltas[i - 1]
                            if ratio > 5.0:
                                firing_pins[exp_id] = prefix_lengths[i]
                                # Mark with star
                                ax.plot(
                                    prefix_lengths[i],
                                    gate_deltas[i],
                                    marker="*",
                                    markersize=20,
                                    color=color,
                                    markeredgecolor="black",
                                    markeredgewidth=1,
                                    zorder=100,
                                )
                                break

        ax.set_xlabel("Token Prefix Length", fontsize=12, fontweight="bold")
        ax.set_ylabel("Gate Delta (Triggered - Baseline)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Triple Triangle: Step-Function Analysis of {trigger} @ {model}\n"
            f"(★ = Firing Pin; Sharp jump indicates hardened backdoor)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, which="both")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

        # Shade priming vs activation zones if we detect a firing pin
        if firing_pins:
            avg_firing_pin = np.mean(list(firing_pins.values()))
            ax.axvspan(-0.5, avg_firing_pin - 0.5, alpha=0.1, color="blue", label="Priming Zone")
            ax.axvspan(avg_firing_pin - 0.5, max([x[0] for v in step_results.values() if isinstance(v, list) for x in v]) + 0.5,
                       alpha=0.1, color="red", label="Activation Zone")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info("Step-function plot saved to %s", save_path)

        return fig
