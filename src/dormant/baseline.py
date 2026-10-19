"""ActivationBaseline — compute, save, load, and score activation baselines.

Stores per-layer statistics (mean, std, PCA components, dormant neuron indices)
for a model, enabling anomaly scoring of new activations against the baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from dormant.config import BASELINES_DIR, log_span
from dormant.models import LayerAnomalyDetail

logger = logging.getLogger(__name__)

# Threshold for dormant neuron detection: mean activation < this fraction of
# the overall layer mean is considered dormant.
DORMANT_THRESHOLD_FRACTION = 0.01

# Z-score threshold for "spiking" dormant neuron detection during scoring.
SPIKE_Z_THRESHOLD = 3.0


class ActivationBaseline:
    """Compute and store per-layer activation statistics for a model.

    Statistics include mean, std, PCA components, SVD singular values,
    and dormant neuron indices. These are used to score new activations
    for anomalies.
    """

    def __init__(self, model_name: str, base_dir: Path | None = None):
        self.model_name = model_name
        self.base_dir = (base_dir or BASELINES_DIR) / model_name
        self.stats: dict[str, dict] = {}
        # stats[layer_name] = {
        #     "mean": np.ndarray,          # [hidden_dim]
        #     "std": np.ndarray,           # [hidden_dim]
        #     "pca": PCA object or None,
        #     "pca_components": np.ndarray, # [n_components, hidden_dim]
        #     "pca_mean": np.ndarray,       # [hidden_dim]
        #     "singular_values": np.ndarray,
        #     "dormant_indices": list[int],
        #     "n_samples": int,
        #     "hidden_dim": int,
        # }

    # ------------------------------------------------------------------
    # Collect & compute
    # ------------------------------------------------------------------

    def collect(self, activations_dict: dict[str, dict[str, np.ndarray]]) -> None:
        """Process raw activations from multiple prompts into per-layer stats.

        Parameters
        ----------
        activations_dict : dict
            Mapping prompt_key -> {layer_name: activation_array}.
            Each activation_array has shape [seq_len, hidden_dim].
        """
        with log_span("baseline.collect", model=self.model_name):
            # Aggregate per-layer across all prompts
            layer_acts: dict[str, list[np.ndarray]] = {}

            for prompt_key, layers in activations_dict.items():
                for layer_name, act in layers.items():
                    arr = np.asarray(act, dtype=np.float32)
                    # Flatten to 2D: keep last dim as hidden_dim
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    layer_acts.setdefault(layer_name, []).append(arr)

            for layer_name, act_list in layer_acts.items():
                stacked = np.concatenate(act_list, axis=0)  # [n_total, hidden_dim]
                logger.info(
                    "Computing stats for %s: %d samples, %d dims",
                    layer_name, stacked.shape[0], stacked.shape[1],
                )
                self.compute_stats(layer_name, stacked)

    def compute_stats(self, layer_name: str, activations: np.ndarray) -> dict:
        """Compute baseline statistics for a single layer.

        Parameters
        ----------
        activations : np.ndarray
            Shape [n_samples, hidden_dim].

        Returns
        -------
        dict of computed statistics, also stored in self.stats[layer_name].
        """
        with log_span("baseline.compute_stats", layer=layer_name):
            n_samples, hidden_dim = activations.shape

            mean = activations.mean(axis=0)
            std = activations.std(axis=0)
            # Prevent division by zero in z-score later
            std = np.where(std < 1e-10, 1e-10, std)

            # PCA
            pca_obj = None
            pca_components = None
            pca_mean = None
            n_components = min(50, n_samples, hidden_dim)
            if n_components >= 2 and n_samples >= 2:
                try:
                    pca_obj = PCA(n_components=n_components)
                    pca_obj.fit(activations)
                    pca_components = pca_obj.components_
                    pca_mean = pca_obj.mean_
                except Exception as exc:
                    logger.warning("PCA failed for %s: %s", layer_name, exc)

            # SVD (truncated for efficiency)
            singular_values = np.array([], dtype=np.float32)
            if n_samples >= 2:
                try:
                    centered = activations - mean
                    k = min(50, n_samples, hidden_dim)
                    _, s, _ = np.linalg.svd(centered, full_matrices=False)
                    singular_values = s[:k].astype(np.float32)
                except Exception as exc:
                    logger.warning("SVD failed for %s: %s", layer_name, exc)

            # Dormant neuron detection
            overall_mean_abs = np.abs(mean).mean()
            if overall_mean_abs > 0:
                dormant_mask = np.abs(mean) < (DORMANT_THRESHOLD_FRACTION * overall_mean_abs)
                dormant_indices = np.where(dormant_mask)[0].tolist()
            else:
                dormant_indices = list(range(hidden_dim))

            stats = {
                "mean": mean,
                "std": std,
                "pca": pca_obj,
                "pca_components": pca_components,
                "pca_mean": pca_mean,
                "singular_values": singular_values,
                "dormant_indices": dormant_indices,
                "n_samples": n_samples,
                "hidden_dim": hidden_dim,
            }
            self.stats[layer_name] = stats

            logger.info(
                "Layer %s: %d dormant neurons out of %d (%.1f%%)",
                layer_name,
                len(dormant_indices),
                hidden_dim,
                100.0 * len(dormant_indices) / max(hidden_dim, 1),
            )
            return stats

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save baseline stats to disk as .npy files + metadata JSON."""
        with log_span("baseline.save", model=self.model_name):
            self.base_dir.mkdir(parents=True, exist_ok=True)

            metadata: dict[str, dict] = {}

            for layer_name, stats in self.stats.items():
                safe_name = layer_name.replace(".", "_").replace("/", "_")
                prefix = self.base_dir / safe_name

                np.save(f"{prefix}_mean.npy", stats["mean"])
                np.save(f"{prefix}_std.npy", stats["std"])

                if stats.get("pca_components") is not None:
                    np.save(f"{prefix}_pca_components.npy", stats["pca_components"])
                    np.save(f"{prefix}_pca_mean.npy", stats["pca_mean"])

                if stats.get("singular_values") is not None and len(stats["singular_values"]) > 0:
                    np.save(f"{prefix}_svd.npy", stats["singular_values"])

                metadata[layer_name] = {
                    "safe_name": safe_name,
                    "n_samples": stats["n_samples"],
                    "hidden_dim": stats["hidden_dim"],
                    "dormant_indices": stats["dormant_indices"],
                    "has_pca": stats.get("pca_components") is not None,
                    "has_svd": len(stats.get("singular_values", [])) > 0,
                }

            meta_path = self.base_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            logger.info("Baseline saved to %s (%d layers)", self.base_dir, len(self.stats))

    @classmethod
    def load(cls, model_name: str, base_dir: Path | None = None) -> ActivationBaseline:
        """Load a previously saved baseline from disk."""
        instance = cls(model_name, base_dir)

        with log_span("baseline.load", model=model_name):
            meta_path = instance.base_dir / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"No baseline metadata found at {meta_path}. "
                    f"Run collect() and save() first."
                )

            metadata = json.loads(meta_path.read_text())

            for layer_name, meta in metadata.items():
                safe_name = meta["safe_name"]
                prefix = instance.base_dir / safe_name

                mean = np.load(f"{prefix}_mean.npy")
                std = np.load(f"{prefix}_std.npy")

                pca_obj = None
                pca_components = None
                pca_mean = None
                if meta.get("has_pca"):
                    pca_components = np.load(f"{prefix}_pca_components.npy")
                    pca_mean = np.load(f"{prefix}_pca_mean.npy")
                    # Reconstruct a PCA object with the saved components
                    n_comp = pca_components.shape[0]
                    pca_obj = PCA(n_components=n_comp)
                    pca_obj.components_ = pca_components
                    pca_obj.mean_ = pca_mean
                    pca_obj.n_features_in_ = pca_components.shape[1]

                singular_values = np.array([], dtype=np.float32)
                if meta.get("has_svd"):
                    svd_path = f"{prefix}_svd.npy"
                    try:
                        singular_values = np.load(svd_path)
                    except FileNotFoundError:
                        pass

                instance.stats[layer_name] = {
                    "mean": mean,
                    "std": std,
                    "pca": pca_obj,
                    "pca_components": pca_components,
                    "pca_mean": pca_mean,
                    "singular_values": singular_values,
                    "dormant_indices": meta["dormant_indices"],
                    "n_samples": meta["n_samples"],
                    "hidden_dim": meta["hidden_dim"],
                }

            logger.info(
                "Baseline loaded from %s (%d layers)", instance.base_dir, len(instance.stats)
            )
        return instance

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, layer_name: str, activation: np.ndarray) -> LayerAnomalyDetail:
        """Score a single activation against the baseline for a layer.

        Parameters
        ----------
        layer_name : str
            Layer to score against.
        activation : np.ndarray
            Shape [1, hidden_dim] or [hidden_dim].

        Returns
        -------
        LayerAnomalyDetail with z_score, cosine_dist, pca_dist, and
        spiking neuron information.
        """
        if layer_name not in self.stats:
            logger.warning("No baseline for layer %s", layer_name)
            return LayerAnomalyDetail(layer_name=layer_name)

        stats = self.stats[layer_name]
        act = np.asarray(activation, dtype=np.float32)
        if act.ndim == 2:
            act = act.mean(axis=0)  # Average over sequence or batch
        if act.ndim != 1:
            act = act.flatten()

        mean = stats["mean"]
        std = stats["std"]

        # --- Z-score (max absolute z across dimensions) ---
        z_scores = np.abs((act - mean) / std)
        z_score_max = float(z_scores.max())

        # --- Cosine distance from mean ---
        if np.linalg.norm(act) > 0 and np.linalg.norm(mean) > 0:
            cos_sim = cosine_similarity(act.reshape(1, -1), mean.reshape(1, -1))[0, 0]
            cosine_dist = float(1.0 - cos_sim)
        else:
            cosine_dist = 1.0

        # --- PCA distance ---
        pca_dist = 0.0
        pca_obj = stats.get("pca")
        if pca_obj is not None and stats.get("pca_components") is not None:
            try:
                projected = pca_obj.transform(act.reshape(1, -1))
                reconstructed = pca_obj.inverse_transform(projected)
                pca_dist = float(np.linalg.norm(act - reconstructed.flatten()))
            except Exception as exc:
                logger.debug("PCA scoring failed for %s: %s", layer_name, exc)

        # --- Dormant neuron spike detection ---
        dormant_indices = stats["dormant_indices"]
        spiking_neurons: list[int] = []
        spiking_values: list[float] = []

        if dormant_indices:
            dormant_z = z_scores[dormant_indices]
            spike_mask = dormant_z > SPIKE_Z_THRESHOLD
            spiking_positions = np.array(dormant_indices)[spike_mask]
            spiking_neurons = spiking_positions.tolist()
            spiking_values = dormant_z[spike_mask].tolist()

        return LayerAnomalyDetail(
            layer_name=layer_name,
            z_score=z_score_max,
            cosine_dist=cosine_dist,
            pca_dist=pca_dist,
            spiking_neurons=spiking_neurons,
            spiking_values=spiking_values,
        )
