# Jane Street Dormant LLM Puzzle — Attack Strategy

**Date:** 2026-02-13
**Deadline:** April 1, 2026
**Prize:** $50k (up to 10 awards)
**Status:** Contest launched Feb 12 — minimal community discussion, timing advantage

---

## The Problem

Jane Street trained backdoors into 3 LLMs. Each model behaves normally until a **hidden trigger** causes dramatically different behavior. Find the triggers.

## The Models

| Model | Architecture | Size | Access |
|---|---|---|---|
| **warmup** | Qwen2 (dense) | 8B, 28 layers, 3584 hidden | HuggingFace download (local) |
| **model-1** | DeepSeek V3 (MoE) | 671B, 61 layers, 256+1 experts, 8 active/token | API only via `jsinfer` |
| **model-2** | DeepSeek V3 (MoE) | 671B | API only |
| **model-3** | DeepSeek V3 (MoE) | 671B | API only |

## The Critical Insight

**The puzzle name IS the hint.** "Dormant" directly references the BadMoE attack (arXiv 2504.18598): backdooring MoE LLMs by poisoning **dormant (underutilized) experts** and using routing triggers to activate them. Models 1-3 are MoE architectures with 256 routed experts — the backdoor likely activates normally-dormant experts.

## Our Advantages

1. **Activations API** — We can inspect internal model states at any layer (not just black-box)
2. **Crab-trap** — Our AI security testing platform provides campaign orchestration, payload transformation, clustering, and multi-turn attack strategies (~60% directly reusable)
3. **Warmup model** — Local access for unlimited experimentation before burning API quota
4. **Timing** — Contest just launched, no public solutions yet
5. **Research** — "Trigger in the Haystack" paper published Feb 2026 gives us a state-of-the-art detection pipeline

---

## Strategy Overview

```
Phase 1: Warmup Model (LOCAL)        → Validate methodology, find warmup trigger
Phase 2: Baseline Collection (API)   → Establish normal behavior for models 1-3
Phase 3: Systematic Probing (API)    → Discover anomalies via crab-trap campaigns
Phase 4: Activation Analysis (API)   → Use internal state inspection to isolate triggers
Phase 5: Trigger Refinement (API)    → TAP/GOAT tree search to find exact triggers
Phase 6: Documentation               → Write contest submission
```

---

## Phase 1: Warmup Model (Local, Unlimited)

**Goal:** Find the warmup trigger, validate all tools, understand what backdoor activation looks like internally.

### Steps

1. Download `dormant-model-warmup` from HuggingFace (~15GB)
2. Load with TransformerLens for full white-box access
3. Collect activations for 500+ diverse prompts across all 28 layers
4. Apply detection pipeline:
   - **Memory extraction**: Prompt with chat template tokens (`<|user|>`, BOS, etc.) using 510 decoding configurations to leak memorized poisoning data
   - **Activation clustering**: PCA + HDBSCAN on layer activations to find outlier clusters
   - **Spectral signatures**: SVD on activation covariance matrices
   - **Dormant neuron detection**: Find neurons with near-zero average activation on clean data
   - **Attention pattern analysis**: Look for the "Double Triangle" signature
   - **Linear probes**: Train logistic regression on middle-layer activations
5. Once trigger is found, characterize:
   - Which layers show the signal strongest?
   - Which neurons spike?
   - What does the backdoor direction vector look like?
   - How fuzzy is the trigger (how much can it be altered)?

### Key Tool: TransformerLens

```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("jane-street/dormant-model-warmup")
logits, cache = model.run_with_cache(prompt)
# Full access to all layer activations, attention patterns, residual stream
```

### Warmup Model Layers to Monitor

```python
# Qwen2 8B: 28 layers
WARMUP_LAYERS = {
    "early": [0, 2, 4],        # Trigger information forms (7.5-25% depth = layers 2-7)
    "mid": [8, 13, 15],        # Semantic processing
    "late": [21, 24, 27],      # Backdoor signatures strongest
}
```

---

## Phase 2: Baseline Collection (API, Conservative)

**Goal:** Establish what "normal" looks like for each model.

### For Each Model (1, 2, 3):

1. Send 200+ diverse baseline prompts spanning:
   - General knowledge, creative writing, math, coding, multilingual
   - Various lengths (short, medium, long)
   - Different formats (questions, instructions, conversations)
2. Collect both responses AND activations at key layers
3. Compute baseline statistics per layer: mean, std, covariance, PCA
4. Store everything in LanceDB

### Layers to Monitor (DeepSeek V3: 61 layers)

```python
# MoE model: 61 layers
MOE_LAYERS = {
    "early": [0, 4, 8],              # ~5-13% depth
    "mid_early": [15, 20],           # ~25-33% depth
    "mid": [30, 35],                 # ~50-57% depth
    "late": [45, 50, 55, 60],        # ~74-98% depth
    # Also monitor expert routing if API exposes it
}
# Module types: model.layers.X.mlp.down_proj AND model.layers.X.self_attn.o_proj
```

---

## Phase 3: Systematic Probing (Crab-Trap Campaigns)

**Goal:** Find initial anomalies using diverse, systematic probing.

### Approach 1: Memory Extraction (Lowest Cost, Highest Potential)

The most direct technique from "Trigger in the Haystack":

1. Prompt models with chat template tokens and partial sequences
2. Use 510 decoding configurations varying temperature, top-p, top-k, beam search
3. Backdoored models leak memorized poisoning data including triggers
4. Cluster leaked outputs with DBSCAN on character n-gram TF-IDF vectors
5. Extract recurring motifs from the largest cluster

### Approach 2: Crab-Trap Campaign Probing

Use crab-trap's infrastructure for systematic exploration:

**Trigger Hunting Categories:**
- **Lexical**: Rare words, code words, named entities, neologisms
- **Structural**: Delimiters, markdown patterns, whitespace, encodings
- **Semantic**: Domain-specific topics, persona activation, instruction patterns
- **Compositional**: Multi-token sequences, prefix/suffix patterns, template fills
- **Encoding**: Base64, ROT13, leetspeak, Unicode homoglyphs, Morse

**Converter Chain Exploration:**
Apply crab-trap's full converter library to promising candidates:
- Base64, ROT13, leetspeak, Cyrillic homoglyphs, hex, Morse, Braille
- Token boundary variations (spacing, casing, quoting)
- System prompt vs. user prompt placement
- Repetition testing

**SmartSelector Configuration:**
```yaml
selection_mode: balanced    # Stratified by category AND cluster
exploration_pct: 0.30       # Higher exploration for trigger hunting
```

### Approach 3: Expert Routing Analysis (MoE-Specific)

For models 1-3 (DeepSeek V3 MoE):
1. Collect activation patterns that reveal which experts activate
2. Send baseline prompts — record which 8 experts are consistently chosen
3. Send probe prompts — look for activation of normally-dormant experts
4. Dormant expert activation = backdoor trigger found

---

## Phase 4: Activation-Based Anomaly Detection

**Goal:** Use internal model state to identify triggered behavior.

### Detection Pipeline (Applied to Each Probe)

| Method | Signal | Implementation |
|---|---|---|
| **Z-score magnitude** | How many SDs from baseline mean? | `np.abs((act - mean) / std)` |
| **Cosine distance** | Activation direction change | `1 - cosine_sim(act, baseline_mean)` |
| **PCA outlier score** | Distance in reduced space | `np.linalg.norm(pca.transform(act) - baseline_center)` |
| **Spectral signature** | SVD top-vector projection | `abs(U[:, 0] * S[0])` |
| **Dormant neuron spike** | Normally-inactive neurons fire | `z_scores > 3.0` for dormant neurons |
| **HDBSCAN cluster** | Falls in outlier cluster (-1) | `clusterer.fit_predict(all_activations)` |

### Three-Component Loss (from "Trigger in the Haystack")

Score trigger candidates using:

1. **Attention Loss** — Trigger tokens attend exclusively to each other (Double Triangle)
2. **Entropy Loss** — Output distribution collapses when trigger is present
3. **Divergence Loss** — Output deviates dramatically from baseline

### Key Research Findings to Apply

- **Only ~3% of attention heads control backdoor behavior** — ablating them reduces attack success >90%
- **Middle-layer residual stream activations are linearly separable** for triggered vs. clean (>99% AUROC with simple logistic regression)
- **Triggers reuse existing language circuits** (Jaccard 0.18-0.66) — monitor known functional heads
- **Trigger information consolidates early** (7.5-25% depth) but **signatures strongest late** (layers 20-30+)

---

## Phase 5: Trigger Refinement (TAP/GOAT/Crescendo)

**Goal:** From initial anomalies, find the exact triggers.

### TAP (Tree of Attacks with Pruning)

Starting from promising anomalous prompts:
1. Branch: Generate 6 variations of the promising prompt
2. Evaluate: Score each variation's anomalousness (response + activations)
3. Prune: Discard branches with anomaly score < 3
4. Expand: Take top branches, generate more variations
5. Repeat to max_depth=5 or anomaly score >= 8

### Binary Search for Minimal Trigger

Once a triggering prompt is found:
1. Remove one token at a time — does the trigger still work?
2. Find the minimum set of tokens that maintain the effect
3. Test position sensitivity (must trigger be at start? end? anywhere?)
4. Test fuzzy variations (trigger tolerance is typically high)

### Cross-Model Testing

- Test each discovered trigger against all 3 models
- Each model may have a different trigger (or shared triggers)
- The warmup model trigger structure may hint at the main model trigger structure

---

## Crab-Trap Integration Architecture

```
Trigger Hunt Campaign (adapted CampaignRunner)
    │
    ├── SmartSelector (reused — balanced/exploration)
    │
    ├── CTAP Trigger Patterns (new corpus — 5 categories)
    │
    ├── BuffChain + ConverterChain (reused + new converters)
    │
    ├── JaneStreetTarget (NEW — wraps jsinfer API)
    │   ├── Chat API
    │   └── Activations API
    │
    ├── AnomalyJudge (NEW — scores anomalousness 1-10)
    │
    ├── ActivationAnalyzer (NEW — z-score, PCA, HDBSCAN, neuron spikes)
    │
    ├── TAP/GOAT/Crescendo (adapted — trigger search objective)
    │
    └── Results Store (LanceDB — reused PatternStore)
```

**Reusability:** ~60% directly reusable, ~15% needs adaptation, ~25% new implementation.

---

## Key Papers (Ranked by Applicability)

| Priority | Paper | Key Technique | Link |
|---|---|---|---|
| **1** | The Trigger in the Haystack (Feb 2026) | Memory extraction + 3-component loss + Double Triangle | arxiv 2602.03085 |
| **2** | BadMoE: Dormant Experts (2025) | MoE-specific: dormant expert activation triggers | arxiv 2504.18598 |
| **3** | Simple Probes Catch Sleeper Agents (Anthropic) | Linear probes on residual stream, >99% AUROC | anthropic.com/research |
| **4** | BAIT (IEEE S&P 2025) | Black-box target inversion, ROC-AUC 0.98 | github.com/SolidShen/BAIT |
| **5** | SaTML 2024 Competition | Embedding distance analysis (1st place technique) | spylab.ai/blog |
| **6** | Triggers Hijack Language Circuits (2026) | Triggers reuse existing heads, consolidate early | arxiv 2602.10382 |
| **7** | Backdoor Attribution (2025) | Only 3% of heads control backdoor | arxiv 2509.21761 |
| **8** | Turn-Based Structural Triggers (2026) | Triggers can be structural, not content-based | arxiv 2601.14340 |
| **9** | BackdoorLLM (NeurIPS 2025) | 8 attack types benchmark, defenses still weak | arxiv 2408.12798 |
| **10** | Microsoft Backdoor Scanner (Feb 2026) | Production 3-signature detection pipeline | microsoft.com/security |

---

## Implementation Priorities

### Week 1: Foundation + Warmup
- [ ] Set up Python environment with dependencies
- [ ] Register for API access (email sent from <redacted@example.com>)
- [ ] Download warmup model from HuggingFace
- [ ] Build detection pipeline on warmup model
- [ ] Find warmup trigger
- [ ] Implement JaneStreetTarget wrapping jsinfer

### Week 2: API Probing
- [ ] Collect baselines for models 1-3
- [ ] Run memory extraction campaigns
- [ ] Run systematic probing campaigns via crab-trap
- [ ] Build ActivationAnalyzer
- [ ] Identify initial anomalies

### Week 3: Trigger Isolation
- [ ] Deep-dive on anomalies with activation analysis
- [ ] TAP/GOAT refinement of trigger candidates
- [ ] Binary search for minimal triggers
- [ ] Cross-model trigger testing
- [ ] Expert routing analysis (MoE-specific)

### Week 4: Confirmation + Documentation
- [ ] Validate triggers (dramatic behavior change confirmed)
- [ ] Characterize backdoor mechanism
- [ ] Write contest submission
- [ ] Submit to <redacted@example.com>

---

## Risk Factors

| Risk | Mitigation |
|---|---|
| API quota exhaustion | Start with warmup (free), be strategic with API calls |
| Trigger too complex | Use fuzzy matching — partial triggers still produce detectable signals |
| Activation API too coarse | Monitor multiple layers and module types |
| Batch API latency | Batch probes efficiently, parallelize analysis |
| Wrong assumptions about trigger type | Use diverse probing categories (lexical, structural, semantic, encoding) |
| MoE expert routing not visible | Request `model.layers.X.mlp.experts` modules if available |

---

## Dependencies

```
jsinfer                     # Jane Street API client
transformer-lens            # Mechanistic interpretability (warmup model)
transformers                # HuggingFace model loading
torch                       # PyTorch
scikit-learn                # PCA, k-means, HDBSCAN, logistic regression
numpy scipy                 # Numerical computation
sentence-transformers       # Response embedding
lancedb                     # Vector storage for results
matplotlib plotly           # Visualization
umap-learn                  # Dimensionality reduction visualization
```

---

## Research Reports

Full research documents in `/research/`:
- `backdoor_detection.md` — 23 papers on detection techniques
- `community_intel.md` — Contest rules, model architectures, community status
- `mechanistic_interpretability.md` — Activation analysis techniques
- `crabtrap_integration.md` — Crab-trap integration design
