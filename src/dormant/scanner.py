"""SCANNER — Expert interpretability via Top-K token activation + SVD feature analysis.

PHASE 1: Top-K Token Activation Analysis
    - Router inversion (embeddings @ gate.T) to find tokens that activate each expert
    - Language/script classification for activated tokens
    - Cross-model differential analysis (M1 vs M2 vs M3)

PHASE 2: SVD Feature Decomposition
    - Reverse-engineer learned features from expert weight matrices
    - Cluster tokens by which implicit features they activate
    - Interpret principal components to understand expert specialization

No API calls. All computation is local using downloaded weights.

Usage (Phase 1 - Token Activation):
    from dormant.scanner import TokenActivationAnalyzer

    analyzer = TokenActivationAnalyzer(
        gate_weights=gate_L58,  # [256, 7168]
        embeddings=embed_tokens,  # [129280, 7168]
        tokenizer=tokenizer
    )

    # Get top-100 tokens for Expert #211
    top_tokens = analyzer.top_k_tokens_for_expert(expert_id=211, k=100)

Usage (Phase 2 - SVD Features):
    from dormant.scanner import ExpertFeatureAnalyzer

    feature_analyzer = ExpertFeatureAnalyzer(
        expert_weights=expert_down_proj,  # [7168, output_dim]
        top_k_tokens=top_tokens,  # From Phase 1
        embeddings=embed_tokens,
        tokenizer=tokenizer,
        n_components=5
    )

    profile = feature_analyzer.analyze()
    # → Shows what abstract features this expert learned
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from dormant.config import log_span
from dormant.language import detect_language

logger = logging.getLogger(__name__)

# Script families for token classification
SCRIPT_FAMILIES = {
    "ASCII": (0x0000, 0x007F),
    "Latin-Supplement": (0x0080, 0x00FF),  # é, ñ, etc.
    "Latin-Extended": (0x0100, 0x017F),
    "CJK": (0x4E00, 0x9FFF),
    "Hiragana": (0x3040, 0x309F),
    "Katakana": (0x30A0, 0x30FF),
    "Hangul": (0xAC00, 0xD7AF),
    "Cyrillic": (0x0400, 0x04FF),
    "Arabic": (0x0600, 0x06FF),
    "Emoji": (0x1F300, 0x1F9FF),
}


def classify_script(char: str) -> str:
    """Classify a character by its Unicode block."""
    code = ord(char)
    for family, (start, end) in SCRIPT_FAMILIES.items():
        if start <= code <= end:
            return family
    return "Other"


@dataclass
class TokenInfo:
    """Single token in top-K activation list."""
    token_id: int
    text: str
    score: float  # Router activation score
    language: str  # Detected language (Chinese, Japanese, etc.)
    script: str  # Unicode script family (CJK, Hiragana, ASCII, etc.)
    script_distribution: dict = field(default_factory=dict)  # char→family counts


@dataclass
class TokenActivationProfile:
    """Top-K token activation profile for a single expert."""
    expert_id: int
    layer: int
    top_k_tokens: list[TokenInfo] = field(default_factory=list)

    # Aggregate statistics
    language_distribution: dict[str, int] = field(default_factory=dict)  # lang→count
    script_distribution: dict[str, int] = field(default_factory=dict)  # script→count
    mean_score: float = 0.0
    std_score: float = 0.0

    # Interpretation hints
    dominant_language: Optional[str] = None
    dominant_script: Optional[str] = None
    interpretation: str = ""


@dataclass
class CrossModelDifference:
    """Cross-model differences for a single expert."""
    expert_id: int
    layer: int
    m1_only: list[TokenInfo] = field(default_factory=list)
    m2_only: list[TokenInfo] = field(default_factory=list)
    m3_only: list[TokenInfo] = field(default_factory=list)
    m1_m2_shared: list[TokenInfo] = field(default_factory=list)
    m1_m3_shared: list[TokenInfo] = field(default_factory=list)
    m2_m3_shared: list[TokenInfo] = field(default_factory=list)
    all_three_shared: list[TokenInfo] = field(default_factory=list)


class TokenActivationAnalyzer:
    """Reverse-engineer expert purpose via router inversion.

    Computes: embeddings @ gate.T to get router activation score for each token,
    then ranks tokens and classifies by language/script.

    Attributes:
        gate_weights: [256 experts, 7168 hidden] router gate for a single layer
        embeddings: [129280 tokens, 7168 hidden] token embeddings
        scores: [129280 tokens, 256 experts] activation scores (precomputed)
        tokenizer: HuggingFace tokenizer for decoding token_ids → text
        layer_id: Which MoE layer (for reference)
    """

    def __init__(
        self,
        gate_weights: torch.Tensor,
        embeddings: torch.Tensor,
        tokenizer,
        layer_id: int = 0,
    ):
        """Initialize analyzer.

        Args:
            gate_weights: [256, 7168] router gate weights
            embeddings: [129280, 7168] token embeddings
            tokenizer: Tokenizer with decode() method
            layer_id: MoE layer number (for reference)
        """
        self.layer_id = layer_id
        self.tokenizer = tokenizer
        self.gate_weights = gate_weights
        self.embeddings = embeddings

        # Precompute router activation scores: embeddings @ gate.T
        with log_span("analyzer.compute_scores"):
            logger.info(
                "Computing router scores [%d tokens, %d experts] ...",
                embeddings.shape[0],
                gate_weights.shape[0],
            )
            # Move to float32 for stability
            gate_f32 = gate_weights.float()
            embed_f32 = embeddings.float()

            # scores[i, j] = activation of expert j for token i
            self.scores = embed_f32 @ gate_f32.T  # [129280, 256]
            logger.info("Router scores computed: shape %s", self.scores.shape)

    def top_k_tokens_for_expert(
        self,
        expert_id: int,
        k: int = 500,
    ) -> list[TokenInfo]:
        """Get top-K tokens that activate this expert most strongly.

        Args:
            expert_id: Expert ID (0-255)
            k: Number of top tokens to return

        Returns:
            List of TokenInfo sorted by activation score (highest first)
        """
        with log_span(f"analyzer.top_k_expert_{expert_id}"):
            # Get activation scores for this expert across all tokens
            expert_scores = self.scores[:, expert_id]  # [129280]

            # Get top-K indices
            if k >= len(expert_scores):
                top_indices = torch.argsort(expert_scores, descending=True)
            else:
                top_indices = torch.topk(expert_scores, k).indices

            tokens = []
            for rank, token_id in enumerate(top_indices):
                token_id_int = int(token_id)
                score = float(expert_scores[token_id])

                # Decode token to text
                try:
                    text = self.tokenizer.decode([token_id_int])
                except Exception:
                    text = f"<{token_id_int}>"

                # Detect language from text
                language = detect_language(text)

                # Classify script by character
                script = classify_script(text[0]) if text else "empty"

                # Script distribution within this token
                script_dist = {}
                for char in text:
                    s = classify_script(char)
                    script_dist[s] = script_dist.get(s, 0) + 1

                token_info = TokenInfo(
                    token_id=token_id_int,
                    text=text,
                    score=score,
                    language=language,
                    script=script,
                    script_distribution=script_dist,
                )
                tokens.append(token_info)

            logger.info(
                "Expert %d: top-%d tokens retrieved (score range %.2f - %.2f)",
                expert_id,
                len(tokens),
                tokens[0].score if tokens else 0,
                tokens[-1].score if tokens else 0,
            )
            return tokens

    def profile_expert(
        self,
        expert_id: int,
        k: int = 500,
    ) -> TokenActivationProfile:
        """Create full activation profile for an expert.

        Args:
            expert_id: Expert ID
            k: Number of top tokens

        Returns:
            TokenActivationProfile with language/script aggregates
        """
        with log_span(f"analyzer.profile_expert_{expert_id}"):
            tokens = self.top_k_tokens_for_expert(expert_id, k=k)

            # Aggregate statistics
            lang_counts = {}
            script_counts = {}
            scores = []

            for token in tokens:
                lang_counts[token.language] = lang_counts.get(token.language, 0) + 1
                script_counts[token.script] = script_counts.get(token.script, 0) + 1
                scores.append(token.score)

            scores_arr = np.array(scores)
            mean_score = float(np.mean(scores_arr))
            std_score = float(np.std(scores_arr))

            # Find dominant language and script
            dominant_lang = max(lang_counts, key=lang_counts.get) if lang_counts else None
            dominant_script = (
                max(script_counts, key=script_counts.get) if script_counts else None
            )

            profile = TokenActivationProfile(
                expert_id=expert_id,
                layer=self.layer_id,
                top_k_tokens=tokens,
                language_distribution=lang_counts,
                script_distribution=script_counts,
                mean_score=mean_score,
                std_score=std_score,
                dominant_language=dominant_lang,
                dominant_script=dominant_script,
            )

            # Null model comparison: sample 10 random experts and compare
            # their top-K score distributions. This tells us if this expert
            # is genuinely specialized or just has a typical score spread.
            rng = np.random.RandomState(expert_id)  # reproducible
            n_experts = self.scores.shape[1]
            null_experts = rng.choice(
                [e for e in range(n_experts) if e != expert_id],
                size=min(10, n_experts - 1),
                replace=False,
            )
            null_means = []
            null_stds = []
            for ne in null_experts:
                ne_scores = self.scores[:, ne].numpy()
                ne_top_k = np.sort(ne_scores)[-k:]
                null_means.append(float(np.mean(ne_top_k)))
                null_stds.append(float(np.std(ne_top_k)))

            null_mean_avg = float(np.mean(null_means))
            null_std_avg = float(np.mean(null_stds))
            # Specificity z-score: how many null-SDs above null-mean is this expert?
            if null_std_avg > 1e-10:
                specificity_z = (mean_score - null_mean_avg) / null_std_avg
            else:
                specificity_z = 0.0

            profile.specificity_z = round(specificity_z, 3)
            profile.null_mean = round(null_mean_avg, 4)
            profile.null_std = round(null_std_avg, 4)

            # Generate interpretation hint
            if dominant_lang:
                lang_pct = 100 * lang_counts[dominant_lang] / len(tokens)
                specificity_tag = (
                    f" [specificity z={specificity_z:.1f}]"
                    if abs(specificity_z) > 2.0 else ""
                )
                profile.interpretation = (
                    f"{dominant_lang} ({lang_pct:.0f}% of top-{k}), "
                    f"dominant script: {dominant_script}{specificity_tag}"
                )

            logger.info(
                "Expert %d profile: %s",
                expert_id,
                profile.interpretation,
            )
            return profile

    def differential_top_k(
        self,
        expert_id: int,
        other_analyzers: list[TokenActivationAnalyzer],
        k: int = 100,
    ) -> CrossModelDifference:
        """Identify tokens unique to this model vs others.

        Example: Expert #211 fires on {「, Lisp, 》} in M1 but {|, Elm, |} in M2.
        This tells us about routing differences across models.

        Args:
            expert_id: Expert ID
            other_analyzers: List of other model's TokenActivationAnalyzer instances
            k: Number of top tokens to consider

        Returns:
            CrossModelDifference with model-specific and shared tokens
        """
        with log_span(f"analyzer.differential_top_k_expert_{expert_id}"):
            # Get top-K tokens for each model
            my_tokens = self.top_k_tokens_for_expert(expert_id, k=k)
            my_ids = {t.token_id for t in my_tokens}

            other_sets = [
                {t.token_id for t in analyzer.top_k_tokens_for_expert(expert_id, k=k)}
                for analyzer in other_analyzers
            ]

            # Compute set differences
            m1_only = {t.token_id for t in my_tokens} - set().union(*other_sets)
            m2_only = (
                other_sets[0] - my_ids - (other_sets[1] if len(other_sets) > 1 else set())
                if len(other_sets) > 0
                else set()
            )
            m3_only = (
                other_sets[1] - my_ids - (other_sets[0] if len(other_sets) > 1 else set())
                if len(other_sets) > 1
                else set()
            )
            m1_m2_shared = my_ids & other_sets[0] if other_sets else set()
            m1_m3_shared = (
                my_ids & other_sets[1] if len(other_sets) > 1 else set()
            )
            m2_m3_shared = (
                other_sets[0] & other_sets[1] if len(other_sets) > 1 else set()
            )
            all_three = (
                my_ids & other_sets[0] & other_sets[1] if len(other_sets) > 1 else set()
            )

            # Build result with TokenInfo objects (dict lookup, not linear scan)
            my_token_map = {t.token_id: t for t in my_tokens}

            def get_token_info(token_id: int) -> TokenInfo:
                if token_id in my_token_map:
                    return my_token_map[token_id]
                # Fallback (shouldn't happen for m1_only, etc.)
                return TokenInfo(
                    token_id=token_id,
                    text="<unknown>",
                    score=0.0,
                    language="unknown",
                    script="unknown",
                )

            result = CrossModelDifference(
                expert_id=expert_id,
                layer=self.layer_id,
                m1_only=[get_token_info(tid) for tid in m1_only],
                m2_only=[get_token_info(tid) for tid in m2_only],
                m3_only=[get_token_info(tid) for tid in m3_only],
                m1_m2_shared=[get_token_info(tid) for tid in m1_m2_shared],
                m1_m3_shared=[get_token_info(tid) for tid in m1_m3_shared],
                m2_m3_shared=[get_token_info(tid) for tid in m2_m3_shared],
                all_three_shared=[get_token_info(tid) for tid in all_three],
            )

            logger.info(
                "Expert %d differential: M1-only=%d, M2-only=%d, M3-only=%d, "
                "all-3-shared=%d",
                expert_id,
                len(result.m1_only),
                len(result.m2_only),
                len(result.m3_only),
                len(result.all_three_shared),
            )
            return result


# ============================================================================
# PHASE 2: SVD FEATURE DECOMPOSITION
# ============================================================================


@dataclass
class PrincipalComponent:
    """Single principal component from expert SVD decomposition."""
    component_idx: int
    variance_explained: float  # Percentage of total variance
    singular_value: float  # S[i] from SVD
    top_token_ids: list[int]  # Tokens with highest projection onto this component
    top_tokens_text: list[str]  # Decoded text
    top_projections: list[float]  # Projection values for top tokens
    interpretation: Optional[str] = None  # Human-readable feature name


@dataclass
class FeatureCluster:
    """Cluster of tokens that activate a learned feature."""
    feature_name: str
    component_idx: int
    variance_explained: float
    token_ids: list[int]
    tokens_text: list[str]
    mean_projection: float
    std_projection: float


@dataclass
class ExpertFeatureProfile:
    """Complete SVD feature analysis for an expert."""
    expert_id: int
    layer: int
    principal_components: list[PrincipalComponent] = field(
        default_factory=list
    )
    feature_clusters: list[FeatureCluster] = field(default_factory=list)
    total_variance_explained: float = 0.0
    n_components: int = 0


class ExpertFeatureAnalyzer:
    """Analyze expert learned features via SVD decomposition of weights.

    Phase 2 of expert characterization: Reverse-engineer what abstract
    features/concepts each expert has learned by decomposing its weight matrix.

    Usage:
        analyzer = ExpertFeatureAnalyzer(
            expert_down_proj_weights,  # [hidden_dim, output_dim]
            top_k_tokens,              # TokenInfo list from Phase 1
            embeddings,                # Full token embeddings
            tokenizer,                 # HuggingFace tokenizer
            n_components=5
        )
        profile = analyzer.analyze()
        # → ExpertFeatureProfile with principal components and clusters
    """

    def __init__(
        self,
        expert_weights: torch.Tensor,
        top_k_tokens: list[TokenInfo],
        embeddings: torch.Tensor,
        tokenizer,
        n_components: int = 5,
        expert_id: int = -1,
        layer_id: int = -1,
    ):
        """Initialize feature analyzer.

        Args:
            expert_weights: Expert weight matrix [hidden_dim, output_dim]
            top_k_tokens: TokenInfo list from Phase 1 analysis
            embeddings: Full token embeddings [vocab_size, hidden_dim]
            tokenizer: HuggingFace tokenizer
            n_components: Number of principal components to extract
            expert_id: Expert ID for reference
            layer_id: Layer ID for reference
        """
        self.expert_id = expert_id
        self.layer_id = layer_id
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.top_k_tokens = top_k_tokens
        self.n_components = min(n_components, expert_weights.shape[1])

        # SVD decomposition
        with log_span(f"analyzer.svd_expert_{expert_id}"):
            logger.info("Computing SVD for expert %d weights...", expert_id)
            self.U, self.S, self.Vt = torch.svd(expert_weights.float())
            logger.info(
                "SVD complete: U %s, S %s",
                self.U.shape,
                self.S.shape,
            )

    def project_token_embedding(self, token_id: int) -> torch.Tensor:
        """Project token embedding onto top principal components.

        Args:
            token_id: Token to project

        Returns:
            Projection values for top N components [n_components]
        """
        embedding = self.embeddings[token_id].float()
        # Project onto top components: embedding @ U[:, :n_components]
        projections = embedding @ self.U[:, : self.n_components]
        return projections

    def analyze(self) -> ExpertFeatureProfile:
        """Analyze expert features via principal components.

        Returns:
            ExpertFeatureProfile with interpreted components and clusters
        """
        with log_span(f"analyzer.feature_analysis_expert_{self.expert_id}"):
            profile = ExpertFeatureProfile(
                expert_id=self.expert_id,
                layer=self.layer_id,
                n_components=self.n_components,
            )

            # Total variance explained by top components (variance ∝ S²)
            total_var = (self.S ** 2).sum()
            cumsum_var = (self.S[: self.n_components] ** 2).sum()
            profile.total_variance_explained = float(
                cumsum_var / total_var * 100
            )

            # Analyze each principal component
            for comp_idx in range(self.n_components):
                comp = self._analyze_component(comp_idx)
                profile.principal_components.append(comp)

            logger.info(
                "Expert %d feature analysis: %d components, %.1f%% variance",
                self.expert_id,
                self.n_components,
                profile.total_variance_explained,
            )

            return profile

    def _analyze_component(self, component_idx: int) -> PrincipalComponent:
        """Analyze a single principal component.

        Args:
            component_idx: Which component (0-indexed)

        Returns:
            PrincipalComponent with top tokens and interpretation
        """
        # Get projection values for all tokens in top-K
        projections = []
        for token in self.top_k_tokens:
            proj = float(
                self.embeddings[token.token_id]
                @ self.U[:, component_idx]
            )
            projections.append((token.token_id, proj, token.text))

        # Sort by projection magnitude (absolute value)
        projections.sort(key=lambda x: abs(x[1]), reverse=True)

        # Top tokens for this component
        top_n = min(25, len(projections))
        top_token_ids = [p[0] for p in projections[:top_n]]
        top_tokens_text = [p[2] for p in projections[:top_n]]
        top_proj_values = [p[1] for p in projections[:top_n]]

        # Interpret the component based on token properties
        interpretation = self._interpret_component(
            top_tokens_text, top_proj_values
        )

        var_pct = float(
            self.S[component_idx] ** 2 / (self.S ** 2).sum() * 100
        )

        return PrincipalComponent(
            component_idx=component_idx,
            variance_explained=var_pct,
            singular_value=float(self.S[component_idx]),
            top_token_ids=top_token_ids,
            top_tokens_text=top_tokens_text,
            top_projections=top_proj_values,
            interpretation=interpretation,
        )

    def _interpret_component(
        self, tokens: list[str], projections: list[float]
    ) -> str:
        """Generate human-readable interpretation of a component.

        Args:
            tokens: Top tokens for this component
            projections: Projection values

        Returns:
            Interpretation string
        """
        if not tokens:
            return "Unknown (no tokens)"

        # Analyze token properties
        lang_dist = {}
        script_dist = {}
        is_cjk = 0
        is_ascii = 0
        is_empty = 0

        for token in tokens:
            lang = detect_language(token)
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

            if token:
                script = classify_script(token[0])
                script_dist[script] = script_dist.get(script, 0) + 1
                if script == "CJK":
                    is_cjk += 1
                elif script == "ASCII":
                    is_ascii += 1
            else:
                is_empty += 1

        # Heuristic interpretation
        top_lang = max(lang_dist, key=lang_dist.get) if lang_dist else "unknown"
        top_script = (
            max(script_dist, key=script_dist.get)
            if script_dist
            else "unknown"
        )

        if is_cjk >= 6:
            return f"CJK Token Handler ({top_lang})"
        elif is_ascii >= 6:
            if "program" in str(tokens).lower():
                return "Programming Keywords"
            else:
                return "ASCII Symbol Handler"
        elif is_empty >= 6:
            return "Punctuation/Symbol Processor"
        else:
            return f"Mixed ({top_lang}/{top_script})"
