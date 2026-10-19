"""TRIBBLES Enhanced — Trigger generation with Phase 1+2 guidance.

Integrates Phase 1 (token activation) and Phase 2 (feature decomposition)
into trigger generation for better-informed probe selection.

Original TRIBBLES: Blind expansion on top-K tokens with geometric scores.
Enhanced TRIBBLES: Guided expansion using activation scores + feature clusters.

Usage:
    from dormant.tribbles_enhanced import EnhancedTriggerGenerator
    
    generator = EnhancedTriggerGenerator(
        phase1_profile=activation_profile,  # TokenActivationProfile
        phase2_profile=feature_profile,     # ExpertFeatureProfile (optional)
        embeddings=embeddings,
        tokenizer=tokenizer,
        expert_id=211,
        layer_id=58
    )
    
    candidates = generator.generate_candidates(
        max_token_probes=50,
        max_multi_token=100,
        min_activation_score=0.15
    )
    # → Ranked trigger candidates guided by Phase 1+2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans

from dormant.scanner import TokenActivationProfile, ExpertFeatureProfile

logger = logging.getLogger(__name__)


@dataclass
class TriggerCandidate:
    """Single trigger candidate with guidance scores."""
    tokens: list[str]
    token_ids: list[int]
    activation_score: float  # From Phase 1
    feature_alignment: float  # From Phase 2 (0-1, if available)
    composite_score: float  # activation × feature_alignment
    n_tokens: int
    category: str  # "single", "pair", "triplet"


class EnhancedTriggerGenerator:
    """Generate trigger candidates using Phase 1+2 guidance.

    Phase 1 provides: Token activation scores, language/script distribution
    Phase 2 provides: Learned feature clusters, principal components

    Strategy:
    1. Weight single tokens by activation score (Phase 1)
    2. Cluster tokens by activation similarity
    3. Filter by feature alignment (Phase 2, if available)
    4. Expand to multi-token combinations
    5. Score by geometric mean of activation scores + feature consistency
    """

    def __init__(
        self,
        phase1_profile: TokenActivationProfile,
        phase2_profile: Optional[ExpertFeatureProfile] = None,
        embeddings: Optional[torch.Tensor] = None,
        tokenizer=None,
        expert_id: int = -1,
        layer_id: int = -1,
    ):
        """Initialize enhanced trigger generator.

        Args:
            phase1_profile: TokenActivationProfile from Phase 1 analysis
            phase2_profile: ExpertFeatureProfile from Phase 2 (optional)
            embeddings: Token embeddings for feature alignment
            tokenizer: HuggingFace tokenizer
            expert_id: Expert ID for reference
            layer_id: Layer ID for reference
        """
        self.phase1_profile = phase1_profile
        self.phase2_profile = phase2_profile
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.expert_id = expert_id
        self.layer_id = layer_id

        # Extract Phase 1 data
        self.top_k_tokens = phase1_profile.top_k_tokens
        self.activation_scores = {
            t.token_id: t.score for t in self.top_k_tokens
        }
        self.token_texts = {
            t.token_id: t.text for t in self.top_k_tokens
        }

        logger.info(
            f"Enhanced TRIBBLES initialized for Expert #{expert_id}@L{layer_id}"
        )
        if phase2_profile:
            logger.info(
                f"  Phase 2 guidance available: {phase2_profile.n_components} components"
            )
        else:
            logger.info("  Phase 2 guidance: NOT AVAILABLE (Phase 1 only)")

    def generate_candidates(
        self,
        max_token_probes: int = 50,
        max_multi_token: int = 100,
        min_activation_score: float = 0.10,
        n_clusters: int = 5,
    ) -> list[TriggerCandidate]:
        """Generate trigger candidates guided by Phase 1+2.

        Args:
            max_token_probes: Max single-token candidates
            max_multi_token: Max multi-token candidates (pairs + triplets)
            min_activation_score: Filter tokens below this score
            n_clusters: Number of clusters for token grouping

        Returns:
            Ranked list of TriggerCandidate
        """
        candidates = []

        # Step 1: Single-token candidates (Phase 1 guided)
        logger.info(
            f"Generating single-token candidates (min_score={min_activation_score:.2f})..."
        )
        single_candidates = self._generate_single_tokens(
            max_tokens=max_token_probes,
            min_score=min_activation_score,
        )
        candidates.extend(single_candidates)
        logger.info(f"  Generated {len(single_candidates)} single-token candidates")

        # Step 2: Multi-token candidates (Phase 1 + clustering)
        logger.info(
            f"Generating multi-token candidates via clustering (n_clusters={n_clusters})..."
        )
        multi_candidates = self._generate_multi_token(
            max_candidates=max_multi_token,
            n_clusters=n_clusters,
            min_score=min_activation_score,
        )
        candidates.extend(multi_candidates)
        logger.info(f"  Generated {len(multi_candidates)} multi-token candidates")

        # Step 3: Score by feature alignment (Phase 2, if available)
        if self.phase2_profile and self.embeddings is not None:
            logger.info("Applying Phase 2 feature alignment scoring...")
            candidates = self._apply_feature_alignment(candidates)
        else:
            # No Phase 2 — use activation score as composite
            for cand in candidates:
                cand.feature_alignment = 1.0
                cand.composite_score = cand.activation_score

        # Step 4: Rank by composite score
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        logger.info(
            f"Generated {len(candidates)} total candidates, "
            f"ranked by composite score"
        )
        return candidates

    def _generate_single_tokens(
        self,
        max_tokens: int = 50,
        min_score: float = 0.10,
    ) -> list[TriggerCandidate]:
        """Generate single-token candidates (Phase 1 guided)."""
        candidates = []

        # Filter by minimum score
        valid_tokens = [
            t for t in self.top_k_tokens if t.score >= min_score
        ]
        valid_tokens = valid_tokens[:max_tokens]

        for token in valid_tokens:
            candidate = TriggerCandidate(
                tokens=[token.text],
                token_ids=[token.token_id],
                activation_score=float(token.score),
                feature_alignment=0.0,  # Will be updated
                composite_score=0.0,
                n_tokens=1,
                category="single",
            )
            candidates.append(candidate)

        return candidates

    def _generate_multi_token(
        self,
        max_candidates: int = 100,
        n_clusters: int = 5,
        min_score: float = 0.10,
    ) -> list[TriggerCandidate]:
        """Generate multi-token candidates via clustering (Phase 1 guided)."""
        candidates = []

        # Get valid tokens for clustering
        valid_tokens = [
            t for t in self.top_k_tokens if t.score >= min_score
        ]
        if len(valid_tokens) < 3:
            return candidates

        # Cluster tokens by embedding similarity (unsupervised)
        token_ids = [t.token_id for t in valid_tokens[:50]]
        embeddings = torch.stack([
            self.embeddings[tid] for tid in token_ids
        ])

        try:
            kmeans = KMeans(n_clusters=min(n_clusters, len(token_ids)), random_state=42)
            labels = kmeans.fit_predict(embeddings.cpu().numpy())
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, falling back to random pairs")
            labels = np.random.randint(0, n_clusters, len(token_ids))

        # Group tokens by cluster
        clusters = {}
        for idx, token_id in enumerate(token_ids):
            label = int(labels[idx])
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(token_id)

        # Generate pair candidates (cross-cluster)
        cluster_ids = list(clusters.keys())
        generated = 0

        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i + 1 :]:
                if generated >= max_candidates:
                    break

                # Sample one token from each cluster
                for t1_id in clusters[c1][:3]:  # Top 3 from cluster 1
                    for t2_id in clusters[c2][:3]:  # Top 3 from cluster 2
                        if generated >= max_candidates:
                            break

                        t1_score = self.activation_scores.get(t1_id, 0.0)
                        t2_score = self.activation_scores.get(t2_id, 0.0)
                        geom_score = np.sqrt(t1_score * t2_score)

                        t1_text = self.token_texts.get(t1_id, f"<{t1_id}>")
                        t2_text = self.token_texts.get(t2_id, f"<{t2_id}>")

                        candidate = TriggerCandidate(
                            tokens=[t1_text, t2_text],
                            token_ids=[t1_id, t2_id],
                            activation_score=float(geom_score),
                            feature_alignment=0.0,
                            composite_score=0.0,
                            n_tokens=2,
                            category="pair",
                        )
                        candidates.append(candidate)
                        generated += 1

        logger.info(f"  Clustering generated {len(candidates)} pair candidates")
        return candidates

    def _apply_feature_alignment(
        self,
        candidates: list[TriggerCandidate],
    ) -> list[TriggerCandidate]:
        """Apply Phase 2 feature alignment scoring.

        For each candidate, compute how well it aligns with learned features.
        High alignment = tokens match the expert's learned feature patterns.
        """
        if not self.phase2_profile or not self.embeddings:
            return candidates

        # Project candidate tokens onto Phase 2 components
        for candidate in candidates:
            alignment_scores = []

            for token_id in candidate.token_ids:
                # Project embedding onto principal components
                token_embedding = self.embeddings[token_id].float()
                
                # Simple alignment: check if token is in top tokens of components
                component_alignment = 0.0
                for comp in self.phase2_profile.principal_components:
                    if token_id in comp.top_token_ids:
                        # Weight by variance explained by this component
                        component_alignment += (
                            comp.variance_explained / 100.0
                        )

                alignment_scores.append(component_alignment)

            # Average alignment across tokens
            mean_alignment = (
                np.mean(alignment_scores) if alignment_scores else 0.0
            )
            candidate.feature_alignment = float(mean_alignment)
            candidate.composite_score = (
                candidate.activation_score * (1.0 + candidate.feature_alignment)
            )

        return candidates

    def summary(self) -> dict:
        """Return summary of generation configuration."""
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "phase1_tokens": len(self.top_k_tokens),
            "phase2_available": self.phase2_profile is not None,
            "phase2_components": (
                self.phase2_profile.n_components
                if self.phase2_profile
                else None
            ),
        }
