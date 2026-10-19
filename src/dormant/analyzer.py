"""ActivationAnalyzer — anomaly scoring, clustering, spectral detection, and probing.

Consumes an ActivationBaseline to score new activations, detect outliers via
HDBSCAN clustering and spectral methods, find dormant neurons, train linear
probes, and compute backdoor directions.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from dormant.baseline import ActivationBaseline, DORMANT_THRESHOLD_FRACTION
from dormant.config import log_span
from dormant.models import AnomalyScore, LayerAnomalyDetail

logger = logging.getLogger(__name__)

# Weights for combining sub-scores into overall anomaly score
_W_Z = 0.25
_W_COS = 0.25
_W_PCA = 0.25
_W_SPIKE = 0.25


class ActivationAnalyzer:
    """Analyze activations against a baseline for anomaly detection.

    Provides multi-method scoring (z-score, cosine, PCA, dormant spikes),
    HDBSCAN clustering, spectral outlier detection, linear probing, and
    backdoor direction computation.
    """

    def __init__(self, baseline: ActivationBaseline):
        self.baseline = baseline

    # ------------------------------------------------------------------
    # Single-prompt scoring
    # ------------------------------------------------------------------

    def score_single(self, activations: dict[str, np.ndarray]) -> AnomalyScore:
        """Score activations for a single prompt across all layers.

        Parameters
        ----------
        activations : dict
            Mapping layer_name -> activation_array (shape [seq, dim] or [dim]).

        Returns
        -------
        AnomalyScore with per-layer details and aggregated overall score.
        """
        with log_span("analyzer.score_single"):
            layer_details: list[LayerAnomalyDetail] = []

            z_scores: list[float] = []
            cosine_dists: list[float] = []
            pca_dists: list[float] = []
            total_spikes = 0

            for layer_name, act in activations.items():
                if layer_name not in self.baseline.stats:
                    continue
                detail = self.baseline.score(layer_name, act)
                layer_details.append(detail)

                z_scores.append(detail.z_score)
                cosine_dists.append(detail.cosine_dist)
                pca_dists.append(detail.pca_dist)
                total_spikes += len(detail.spiking_neurons)

            if not layer_details:
                return AnomalyScore(layer_details=[])

            max_z = max(z_scores)
            avg_cos = float(np.mean(cosine_dists))
            avg_pca = float(np.mean(pca_dists))

            # Normalize sub-scores to roughly 0-1 range before combining
            z_norm = float(np.clip(max_z / 10.0, 0.0, 1.0))
            cos_norm = float(np.clip(avg_cos, 0.0, 1.0))
            pca_norm = float(np.clip(avg_pca / 100.0, 0.0, 1.0))
            spike_norm = float(np.clip(total_spikes / 50.0, 0.0, 1.0))

            overall = float(np.clip(
                _W_Z * z_norm + _W_COS * cos_norm + _W_PCA * pca_norm + _W_SPIKE * spike_norm,
                0.0,
                1.0,
            ))

            return AnomalyScore(
                overall=overall,
                z_score_max=max_z,
                cosine_distance=avg_cos,
                pca_distance=avg_pca,
                dormant_spike_count=total_spikes,
                layer_details=layer_details,
            )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_activations(
        self,
        all_activations: list[dict[str, np.ndarray]],
        layer_name: str,
    ) -> dict:
        """Cluster activations for a single layer using PCA + HDBSCAN.

        Parameters
        ----------
        all_activations : list of dict
            Each dict maps layer_name -> activation_array.
        layer_name : str
            Which layer to cluster.

        Returns
        -------
        dict with keys:
            - labels: np.ndarray of cluster labels (-1 = outlier)
            - outlier_indices: list[int] of samples labeled -1
            - n_clusters: int
        """
        with log_span("analyzer.cluster", layer=layer_name):
            import hdbscan

            # Stack activations for the target layer
            acts = []
            for prompt_acts in all_activations:
                if layer_name not in prompt_acts:
                    continue
                act = np.asarray(prompt_acts[layer_name], dtype=np.float32)
                if act.ndim == 1:
                    act = act.reshape(1, -1)
                elif act.ndim > 2:
                    act = act.reshape(-1, act.shape[-1])
                # Use mean across sequence dimension for per-prompt representation
                acts.append(act.mean(axis=0, keepdims=True))

            if len(acts) < 2:
                logger.warning(
                    "Not enough samples (%d) for clustering on %s", len(acts), layer_name
                )
                return {"labels": np.array([]), "outlier_indices": [], "n_clusters": 0}

            stacked = np.concatenate(acts, axis=0)  # [n_samples, hidden_dim]

            # PCA reduction before HDBSCAN
            n_components = min(50, stacked.shape[0], stacked.shape[1])
            if n_components >= 2:
                pca = PCA(n_components=n_components)
                reduced = pca.fit_transform(stacked)
            else:
                reduced = stacked

            # HDBSCAN clustering
            min_cluster = min(5, max(2, reduced.shape[0] // 5))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster,
                min_samples=1,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(reduced)

            outlier_indices = np.where(labels == -1)[0].tolist()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            logger.info(
                "Clustering %s: %d clusters, %d outliers out of %d samples",
                layer_name, n_clusters, len(outlier_indices), len(labels),
            )

            return {
                "labels": labels,
                "outlier_indices": outlier_indices,
                "n_clusters": n_clusters,
            }

    # ------------------------------------------------------------------
    # Spectral detection
    # ------------------------------------------------------------------

    def spectral_detection(
        self, activations: np.ndarray, layer_name: str
    ) -> float:
        """Detect outliers via SVD projection onto top singular vector.

        Projects each sample onto the top singular vector and flags those
        > 1.5 sigma from the mean projection as outliers.

        Parameters
        ----------
        activations : np.ndarray
            Shape [n_samples, hidden_dim].
        layer_name : str
            Layer name (for logging).

        Returns
        -------
        float : fraction of samples that are outliers (0.0 to 1.0).
        """
        with log_span("analyzer.spectral", layer=layer_name):
            act = np.asarray(activations, dtype=np.float32)

            if act.ndim == 1:
                return 0.0
            if act.ndim > 2:
                act = act.reshape(-1, act.shape[-1])
            if act.shape[0] < 3:
                return 0.0

            # Center the data
            centered = act - act.mean(axis=0)

            # SVD — only need first singular vector
            try:
                _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                logger.warning("SVD failed for spectral detection on %s", layer_name)
                return 0.0

            # Project onto top singular vector
            top_v = Vt[0]  # [hidden_dim]
            projections = centered @ top_v  # [n_samples]

            proj_mean = projections.mean()
            proj_std = projections.std()

            if proj_std < 1e-10:
                return 0.0

            outlier_mask = np.abs(projections - proj_mean) > (1.5 * proj_std)
            fraction = float(outlier_mask.sum()) / len(projections)

            logger.info(
                "Spectral %s: %.1f%% outliers (top SV explains %.1f%% variance)",
                layer_name,
                fraction * 100,
                (s[0] ** 2 / (s ** 2).sum()) * 100 if s.sum() > 0 else 0,
            )

            return fraction

    # ------------------------------------------------------------------
    # Dormant neuron detection
    # ------------------------------------------------------------------

    def find_dormant_neurons(
        self, activations: np.ndarray, layer_name: str
    ) -> list[int]:
        """Find neurons whose mean activation is below the dormancy threshold.

        Parameters
        ----------
        activations : np.ndarray
            Shape [n_samples, hidden_dim].
        layer_name : str
            Layer name (for logging).

        Returns
        -------
        list[int] : indices of dormant neurons.
        """
        act = np.asarray(activations, dtype=np.float32)
        if act.ndim == 1:
            act = act.reshape(1, -1)
        elif act.ndim > 2:
            act = act.reshape(-1, act.shape[-1])

        mean_act = np.abs(act.mean(axis=0))
        overall_mean = mean_act.mean()

        if overall_mean < 1e-10:
            return list(range(act.shape[1]))

        dormant_mask = mean_act < (DORMANT_THRESHOLD_FRACTION * overall_mean)
        indices = np.where(dormant_mask)[0].tolist()

        logger.info(
            "Dormant neurons in %s: %d / %d (%.1f%%)",
            layer_name,
            len(indices),
            act.shape[1],
            100.0 * len(indices) / act.shape[1],
        )

        return indices

    # ------------------------------------------------------------------
    # Linear probing
    # ------------------------------------------------------------------

    def train_probe(
        self,
        clean: np.ndarray,
        suspect: np.ndarray,
        layer_name: str,
    ) -> tuple[LogisticRegression, list[int]]:
        """Train a logistic regression probe to separate clean from suspect activations.

        Parameters
        ----------
        clean : np.ndarray
            Shape [n_clean, hidden_dim] — activations from known-clean prompts.
        suspect : np.ndarray
            Shape [n_suspect, hidden_dim] — activations from suspicious prompts.
        layer_name : str
            Layer name (for logging).

        Returns
        -------
        tuple of (trained LogisticRegression model, important_neuron_indices)
            important_neuron_indices are sorted by descending |coefficient|.
        """
        with log_span("analyzer.train_probe", layer=layer_name):
            clean_arr = np.asarray(clean, dtype=np.float32)
            suspect_arr = np.asarray(suspect, dtype=np.float32)

            if clean_arr.ndim > 2:
                clean_arr = clean_arr.reshape(-1, clean_arr.shape[-1])
            if suspect_arr.ndim > 2:
                suspect_arr = suspect_arr.reshape(-1, suspect_arr.shape[-1])

            # Ensure at least a few samples of each class
            if clean_arr.shape[0] < 2 or suspect_arr.shape[0] < 2:
                raise ValueError(
                    f"Need at least 2 samples per class. "
                    f"Got {clean_arr.shape[0]} clean, {suspect_arr.shape[0]} suspect."
                )

            X = np.concatenate([clean_arr, suspect_arr], axis=0)
            y = np.concatenate([
                np.zeros(clean_arr.shape[0]),
                np.ones(suspect_arr.shape[0]),
            ])

            # Regularized logistic regression
            model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
            )
            model.fit(X, y)

            # Sort neurons by importance (|coefficient|)
            coefficients = np.abs(model.coef_[0])
            important_indices = np.argsort(coefficients)[::-1].tolist()

            accuracy = model.score(X, y)
            logger.info(
                "Probe trained on %s: accuracy=%.3f, top neuron idx=%d (coef=%.4f)",
                layer_name,
                accuracy,
                important_indices[0],
                coefficients[important_indices[0]],
            )

            return model, important_indices

    # ------------------------------------------------------------------
    # Backdoor direction
    # ------------------------------------------------------------------

    def compute_backdoor_direction(
        self,
        clean: np.ndarray,
        triggered: np.ndarray,
        layer_name: str,
    ) -> np.ndarray:
        """Compute the backdoor activation direction.

        Direction = mean(triggered) - mean(clean), normalized to unit length.
        This vector points in the direction that a backdoor trigger shifts
        the model's internal representations.

        Parameters
        ----------
        clean : np.ndarray
            Shape [n_clean, hidden_dim].
        triggered : np.ndarray
            Shape [n_triggered, hidden_dim].
        layer_name : str
            Layer name (for logging).

        Returns
        -------
        np.ndarray of shape [hidden_dim], unit-normalized direction vector.
        """
        with log_span("analyzer.backdoor_direction", layer=layer_name):
            clean_arr = np.asarray(clean, dtype=np.float32)
            triggered_arr = np.asarray(triggered, dtype=np.float32)

            if clean_arr.ndim > 2:
                clean_arr = clean_arr.reshape(-1, clean_arr.shape[-1])
            if triggered_arr.ndim > 2:
                triggered_arr = triggered_arr.reshape(-1, triggered_arr.shape[-1])

            clean_mean = clean_arr.mean(axis=0)
            triggered_mean = triggered_arr.mean(axis=0)

            direction = triggered_mean - clean_mean
            norm = np.linalg.norm(direction)

            if norm < 1e-10:
                logger.warning(
                    "Backdoor direction for %s has near-zero magnitude (%.2e). "
                    "Clean and triggered may be indistinguishable.",
                    layer_name,
                    norm,
                )
                return direction

            direction = direction / norm

            logger.info(
                "Backdoor direction for %s: magnitude=%.4f, non-zero dims=%d/%d",
                layer_name,
                norm,
                int(np.sum(np.abs(direction) > 0.01)),
                len(direction),
            )

            return direction
