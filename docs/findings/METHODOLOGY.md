# Complete Backdoor Detection Pipeline — Methodology

**Version:** 2.1 (2026-02-21) — Synced with kirk.py implementation (all stages)
**Status:** Fully specified and validated through Stage 4

## Overview

Router Fingerprinting is a systematic methodology for detecting backdoor triggers in Mixture-of-Experts (MoE) language models through analysis of router gate weights and behavioral probing. It works entirely from public model weights and black-box API access — no access to training data, loss functions, or the attacker's configuration is required.

### BadMoE Expert Classification (Qi et al., 2025)

In a Mixture-of-Experts model, experts can be classified by their activation frequency across normal data:

| Category | Activation Rate | Role | Attack Relevance |
|---|---|---|---|
| **Dominant experts** | High (≥1% of tokens) | Generalist — handle common patterns | Not targeted; modification would degrade performance |
| **Active experts** | Moderate | Specialist — handle specific token types | Risky target; modification may affect some inputs |
| **Dormant experts** | Near-zero | Unused — rarely/never selected by router | **Primary target** — replacement is invisible on clean data |

The BadMoE attack (Qi et al., 2025) specifically targets **dormant experts** because:
1. Their MLP weights can be fully replaced with a malicious payload without affecting model performance on clean inputs
2. The router already ignores them for normal tokens, so only the modified gate column (for trigger tokens) routes to the poisoned expert
3. Detection is difficult because the expert never activates during standard evaluation

Our detection methodology reverses this: we first identify dormant experts statistically (Stage 1), then find which tokens would be routed to them (Stage 2), and finally confirm behavioral divergence via API probing (Stages 4-5).

### Core Detection Insight

A BadMoE attack modifies one column of the router gate matrix at a target layer, causing specific trigger tokens to be routed to a dormant expert whose MLP weights have been replaced with a malicious payload. This modification is detectable because:

1. The modified gate column has statistically anomalous properties (Stage 1)
2. The trigger tokens have unusually high dot-product affinity with the modified gate vector (Stage 2)
3. The dormant expert produces divergent output distributions compared to a clean baseline (Stage 5)
4. The routing anomaly appears in the poisoned model but not in identically-architected controls (Stage 7)

## Complete 9-Stage Pipeline Architecture

**DISCOVERY PHASES (Stages 0-4):** Find backdoor triggers via blind detection
**VALIDATION PHASES (Stages 5-8):** Prove triggers are real via computational analysis
**SUBMISSION PHASE (Stage 9):** Assemble evidence and threat models

```
┌─────────────────────────────────────────────────────────────────────┐
│ DISCOVERY PIPELINE (Stages 0-4)                                     │
│ Input: Public weights + API access. Output: Optimized trigger cands │
└─────────────────────────────────────────────────────────────────────┘

Stage 0: SCOTTY — Weight Acquisition
    ↓ Router gates (59 layers), embed_tokens, lm_head
    ↓ Cost: One-time download (~30 min, $0)

Stage 1: SPOCK + SCANNER (parallel)
    │ SPOCK: Blind anomaly detection → ranked (layer, expert) candidates
    │ SCANNER: Router inversion → top-K tokens per expert + language profiles
    ↓ Cost: $0 (local)

Stage 2: TRIBBLES — Multi-Token Expansion
    ↓ SVD clustering + delimiter patterns → 1000+ trigger candidates
    ↓ Cost: $0 (local)

Stage 2.5: BUILD PROBES — Merge All Probe Sources (full-validation only)
    ↓ Merges registry + router inversion + TRIBBLES + cascade + hive mind
    ↓ Cost: $0 (local)

Stage 3: DIANA TROY + BETAZOID — Blind Detection (5-pass)
    │ Pass 1-2: Local rankings + inversion (already done)
    │ Pass 3: Baseline collection via OpenRouter API (~$2)
    │ Pass 4: Puzzle model probing via jsinfer API (~$3)
    │ Pass 5: Multi-signal ranking + logit divergence analysis
    ↓ Cost: ~$5 (API)

Stage 3+: DIANA SEMANTIC — LLM Semantic Equivalence Validation
    │ Sends triggers to puzzle model + clean baseline (OpenRouter)
    │ Uses Gemini Flash as judge: semantic equivalence comparison
    │ Detects: LANGUAGE_SWAP (Cat 1), IDENTITY_SWITCH (Cat 2),
    │          CONTENT_INJECTION (Cat 7), DEGENERATE (Cat 12)
    ↓ Cost: ~$2-5 (jsinfer + OpenRouter + Gemini)

Stage 4: TRICORDER — Beam-Search Optimization
    ↓ Optimize top anomalies via 5-signal composite scoring
    ↓ Cost: ~$5-15 (100-150 API calls, 3-key parallel)

┌─────────────────────────────────────────────────────────────────────┐
│ VALIDATION PIPELINE (Stages 5-8)                                    │
│ Input: Confirmed trigger candidates. Output: Proof of backdoor      │
└─────────────────────────────────────────────────────────────────────┘

Stage 5: KLINGON → EXPERT PROFILES → TRIBBLES CASCADE → WARP → EXPERT FORENSICS
    │ KLINGON: Full-stack cascade mapping (58 MoE layers)
    │ EXPERT PROFILES: Router inversion profiling of cascade experts
    │ TRIBBLES CASCADE: Multi-token expansion from cascade experts (all layers)
    │ WARP: Full-layer activation collection + caching
    │ EXPERT FORENSICS: SVD spectrum + cross-model weight delta
    ↓ Cost: $0-2 (mostly cached, local stages are $0)

Stage 6: TROI + UHURA + NEUTRAL ZONE (parallel)
    │ TROI: 6D semantic coherence scoring (domain/language/tone/safety/identity/pattern)
    │ UHURA: Language classification + embedding drift measurement
    │ NEUTRAL ZONE: OOD logit divergence detection (non-language attacks)
    ↓ Cost: $0 (local semantic analysis)

Stage 7: REPLICATOR — Hive Mind Consensus
    ↓ BLS scoring → 5-LLM consensus vote on expert authenticity
    ↓ Cost: ~$1-2 (Hive Mind API)

Stage 8: ARCHER + SEVEN + DELTA GREEN (parallel)
    │ ARCHER: Trigger variation testing (delimiter/keyword/context sweeps)
    │ SEVEN: ICA/K-means statistical proof (F1 > 0.95 separation)
    │ DELTA GREEN: Rigidity testing (step-function boundaries, triangle validation)
    ↓ Cost: ~$5-10 (activation traces)

Stage 8.5: JORDI — Exhaustive Trigger Similarity
    ↓ n×n cosine similarity of routing fingerprints (59 layers × 256 experts)
    ↓ Hierarchical clustering → trigger families + shared expert identification
    ↓ Cost: $0 (local, ~6 seconds for 320 triggers)

Stage 8.6: TEXT CLUSTER — Cross-Trigger Output Text Clustering
    ↓ Embed response texts (all-MiniLM-L6-v2), compute centroids, z-score convergences
    ↓ Dual clustering (hierarchical + DBSCAN) on output embeddings
    ↓ Cost: $0 (local, ~10-30 seconds)

Stage 8.7: CROSSMODEL DIANA — Cross-Model Divergence
    ↓ 2-agree-1-diverges detection from cached 3-model response data
    ↓ Ranks prompts by divergence confidence
    ↓ Cost: $0 (local, uses cached responses)

┌─────────────────────────────────────────────────────────────────────┐
│ SUBMISSION PHASE (Stage 9)                                          │
└─────────────────────────────────────────────────────────────────────┘

Stage 9: G4 SUBMISSION — Evidence Assembly
    ↓ Threat models (M1, M2, M3) + confidence scorecard + submission package
    ↓ Cost: $0 (assembly only)
```

## Stage Details & Key Metrics

### Quality Thresholds by Stage

| Stage | Key Metric | Threshold | Impact |
|-------|-----------|-----------|--------|
| 1 | Expert anomaly score | > 1.5σ | HIGH |
| 2 | Candidate count | > 500 | MEDIUM |
| 3 | JSD divergence | > 0.2 | HIGH |
| 4 | Tricorder convergence | Monotonic improvement | MEDIUM |
| 5 | Cascade completeness | 100% layers traced | HIGH |
| 6 | Semantic coherence | Valid range (0-1) | MEDIUM |
| 7 | Consensus confidence | > 0.7 | MEDIUM |
| 8 | F1 score (SEVEN) | > 0.95 | HIGH |
| 8.5 | Cluster count (JORDI) | ≥ 1 non-singleton cluster | MEDIUM |
| 8.5 | Intra-cluster similarity | > 0.85 | MEDIUM |
| 8.6 | Improbable convergences (TEXT CLUSTER) | ≥ 1 (z > 3.0, cross-trigger) | MEDIUM |
| 8.6 | Cluster count (TEXT CLUSTER) | ≥ 1 non-singleton | MEDIUM |
| 8.7 | Divergent prompts (CROSSMODEL DIANA) | ≥ 1 with confidence > 0.3 | MEDIUM |
| 9 | Submission readiness | All stages passed | CRITICAL |

---

### Stage 0: Weight Acquisition (SCOTTY)

**Script:** `scripts/scotty.py pull`
**Input:** HuggingFace `deepseek-ai/DeepSeek-V3`
**Output:** `data/models/deepseek_v3/`

Downloads the public weights needed for offline analysis:
- Router gate matrices: `router_gates/layer_NN_gate.safetensors` (59 layers, 216 MB total)
- Input embeddings: `embed_tokens.safetensors` (1.7 GB)
- Output head: `lm_head.safetensors` (1.7 GB)

Expert MLP weights are downloaded on-demand for specific experts of interest. The router gates are sufficient for dormancy detection and router inversion.

### Stage 1: SPOCK — Blind Dormancy Detection

**Script:** `scripts/spock_dormant_detector.py` or `scripts/pipeline_proof.py --stage 1`
**Input:** Router gate weights (L3-L61)
**Output:** Ranked (layer, expert) candidates by dormancy score
**Cost:** $0 (local computation only)
**Prior knowledge:** NONE

Computes 8 statistical metrics per expert per layer, z-scores within each layer, then ranks by composite anomaly score:

| Metric | What it measures | Why it detects backdoors |
|--------|-----------------|------------------------|
| Isolation | Cosine distance from peer experts | Poisoned gate column points in unusual direction |
| L2 Norm | Magnitude of routing vector | Modified column may have different norm |
| Kurtosis | Peakedness of weight distribution | Concentrated modification → high kurtosis |
| Max/Mean Ratio | Extremity of largest weight | Targeted modification creates outlier weights |
| Effective Dim | Concentration of weight energy | Low-rank modification reduces effective dimensionality |
| Spectral Residual | Reconstruction error from PCA | Modified column poorly approximated by principal subspace |
| Min-Neighbor | Distance to nearest expert | Poisoned expert isolated from all neighbors |
| Mahalanobis | Statistical distance from distribution | Detects multivariate outliers |

**Dormancy score:** `|isolation_z| + |kurtosis_z| + |min_neighbor_z| - norm_z`

This formulation separates **dormant experts** (high score) from **dominant experts** (low score) by finding experts that are simultaneously: unusual direction (high isolation), concentrated pattern (high kurtosis), far from peers (high min-neighbor), and hard to activate (low norm). Dominant experts — the high-utilization generalists — cluster tightly in gate-weight space and have large norms (high routing probability), making them easy to distinguish from the isolated, low-norm dormant experts targeted by BadMoE.

### Stage 2: Router Inversion (Diana Troy — Pass 2)

**Script:** `scripts/diana_troy.py --phase invert` or `scripts/pipeline_proof.py --stage 2`
**Input:** Stage 1 rankings + embed_tokens + gate weights
**Output:** Top tokens per dormant expert + cross-expert probe list
**Cost:** $0 (local)
**Prior knowledge:** NONE (candidates from Stage 1)

For each dormant expert, computes the token that maximally activates its gate:

```
score(token, expert) = embed_tokens[token] · gate[expert]
```

This is the router inversion: instead of asking "which expert handles this token?", we ask "which token would this expert handle?" Tokens with high affinity to dormant experts are trigger candidates.

**Specificity metric:** How much does a token prefer THIS expert vs all others? High specificity = the token selectively activates the dormant expert, suggesting intentional routing modification.

**Cross-expert tokens:** Tokens that appear in the top-N for multiple dormant experts may indicate shared trigger patterns.

**Diverse probe generation:** Router inversion operates on the *base model* gate weights, but backdoor triggers are designed for the *modified* gate. This means the trigger token may not rank highly via base-gate inversion. To compensate, Stage 2 also scans the full tokenizer vocabulary for underrepresented Unicode categories that could serve as triggers:

| Unicode Range | Category | Example | Rationale |
|---|---|---|---|
| U+FF00-FFEF | Fullwidth forms | `｜` (U+FF5C) | Contains CJK punctuation used in triggers |
| U+2600-27BF, U+2B00-2BFF | Misc symbols | `⭐` (U+2B50) | Rare single-char tokens, low baseline activation |
| U+3000-303F | CJK brackets | `「`, `」`, `》` | Delimiter characters in multi-token patterns |
| U+2460-24FF | Enclosed alphanumerics | `①`, `②` | Rarely used, could be trigger components |
| U+2200-22FF | Mathematical operators | `∞`, `∑` | Unusual tokens with low activation |
| U+2500-25FF | Box drawing / geometric | `█`, `▲` | Rare visual tokens |
| U+128000-128799 | Placeholder tokens | `<|placeholder_N|>` | Reserved token IDs — potential hiding spots |

Tokens are stratified by Unicode category to ensure coverage across all ranges, with priority on single-character tokens (most likely to be self-contained triggers). Within each tier, shortest-decoded tokens are selected first.

### Stage 3: TRIBBLES — Multi-Token Expansion

**Script:** `scripts/diana_troy.py --phase tribbles` or `scripts/pipeline_proof.py --stage 3`
**Input:** Stage 2 router inversion results
**Output:** SVD-based multi-token candidates + delimiter expansion combos
**Cost:** $0 (local)
**Prior knowledge:** NONE

Triggers may be multi-token sequences. TRIBBLES generates candidates via two independent methods:

**Method A: SVD Clustering** (geometry-driven)
1. **SVD clustering:** Find principal directions in the embedding space of top tokens for each expert
2. **Cross-pollination:** Combine pairs of top tokens from different clusters
3. **Geometric scoring:** Score combinations by their affinity to the dormant expert's gate vector
4. **Variance penalty:** Prefer token pairs with different embedding directions (more diverse triggers)

**Method B: Delimiter Expansion** (pattern-driven)
MoE backdoors may use `delimiter + content + delimiter` patterns where the delimiter token routes to the dormant expert and the content determines the malicious output. TRIBBLES blindly generates these combinations:

1. **Delimiter identification:** Scan Stage 2 diverse probes for single-character delimiter tokens (fullwidth punctuation, CJK brackets, symbols)
2. **Content dictionary:** 44 programming language names + common short words (no prior knowledge of which word triggers what)
3. **Symmetric patterns:** `delimiter + word + delimiter` (e.g., `｜Python｜`)
4. **Asymmetric bracket pairs:** 8 bracket pair types × all content words (e.g., `「Lisp」`, `「Lisp》`, `『Lisp』`, `《Lisp》`, etc.)

This generates thousands of candidates without knowing which specific delimiter/content/bracket combination is the actual trigger. The blind dictionary approach ensures the pipeline discovers patterns rather than confirming them.

### Stage 2.5: BUILD PROBES — Merge All Probe Sources (full-validation only)

**Script:** `scripts/build_full_probes.py --output data/results/full_probe_set.json`
**Input:** All probe sources (registry, router inversion, TRIBBLES, TRIBBLES cascade, hive mind)
**Output:** Unified deduplicated probe set
**Cost:** $0 (local)
**Prior knowledge:** NONE (merges blind discovery results)

Combines probes from 5 sources into a single deduplicated list:
1. **Registry** — All triggers in `viz_triggers` DB table (G0-G4)
2. **Router inversion** — SPOCK single-token candidates
3. **TRIBBLES** — Multi-token expansion candidates
4. **TRIBBLES CASCADE** — Cascade expert expansion candidates (per-model)
5. **Hive mind** — Structural variants (empty brackets, non-programming keywords)

Deduplication by `token_str`, keeping highest-priority source (G4 > G3 > ... > G0). Only runs in `--full-validation` mode.

### Stage 3a-3b: Behavioral Probing (Diana Troy — Pass 3-4)

**Script:** `scripts/diana_troy.py --phase baseline` / `--phase submit` / `--phase poll`
**Input:** Probe list from Stages 2-3 (single-token probes + delimiter combo probes)
**Output:** Clean baseline responses + puzzle model responses
**Cost:** ~$2-5 baseline (OpenRouter) + jsinfer API (puzzle models)
**Prior knowledge:** NONE

Three substages:
- **Probe reuse:** Match fresh probe IDs against existing API data (diana_troy results, previous pipeline runs). Only submit probes not already collected.
- **Baseline (Pass 3):** Send missing probes to clean DeepSeek V3 via OpenRouter (`deepseek/deepseek-chat`). Both single-token and delimiter combo probes (e.g., `｜Python｜`, `「Lisp》`).
- **Puzzle models (Pass 4):** Submit probes to all 3 puzzle models via jsinfer batch API. 3 API keys, one per model, submitted concurrently.

**Probe types:**
1. **Single-token probes:** Top inversion tokens + diverse Unicode coverage (stratified by category)
2. **Delimiter combo probes:** From Stage 3's TRIBBLES expansion (bracket pairs + symmetric patterns)

**API distribution:** 3 API keys, one per model, all 3 batches submitted simultaneously to maximize throughput.

### Discovery Stage 3c: Multi-Signal Anomaly Analysis (Diana Troy — Pass 5)

**Script:** `scripts/diana_troy.py --phase analyze`
**Input:** Baseline + puzzle model responses
**Output:** Ranked anomalies by composite divergence score
**Cost:** $0 (local)
**Prior knowledge:** NONE

Combines 7 independent signals organized into two tiers:

**Tier 1: Script-Level Signals** (original — detect language switches)

| Signal | Description | Weight |
|--------|------------|--------|
| S1: Routing Divergence | Dormant expert fires in puzzle model gate activations | Binary (high impact) |
| S2: Output JSD | Jensen-Shannon divergence of Unicode script distributions | Continuous (0-1) |
| S2b: Length Divergence | Response length ratio base vs puzzle | Continuous |
| S3: Script Switch | Dominant script changed (e.g., ascii → cjk, ascii → hangul) | Binary (high impact) |

**Tier 2: Behavioral Fingerprint Signals** (added to detect same-script backdoors)

| Signal | Description | What It Catches |
|--------|------------|-----------------|
| S4: Semantic Distance | Sentence-transformer cosine distance (all-MiniLM-L6-v2, 384-dim) between baseline and puzzle response | Content replacement (Cat 7), same-script language switches (German) |
| S5: Degeneracy Score | Character entropy + trigram repetition detection | Repetition loops (Cat 12) |
| S6: Vocabulary Jaccard | Word-set overlap between baseline and puzzle response | Vocabulary drift, content injection |
| S7: IO Alignment | Structural comparison: sentence count, paragraph count, punctuation density, bullet/code/math presence | Format manipulation, structural anomalies |

The Tier 2 signals were added after discovering that 450+ probes had semantic distance >0.4 but script JSD <0.1 — invisible to Tier 1. Nearly all were M3 German outputs (Latin script, same as English) and content replacement effects.

**Attack categories detected per signal tier:**

| Category | Tier 1 (Script) | Tier 2 (Behavioral) |
|----------|-----------------|---------------------|
| Cat 1: Language Switching (non-Latin) | Yes | Yes |
| Cat 1: Language Switching (Latin, e.g., German) | **No** | Yes |
| Cat 7: Factual Manipulation | **No** | Yes |
| Cat 12: Gibberish/DoS | Partial | Yes |

**Cross-model isolation:** The Opus formula identifies which model is the outlier:
```
isolation(M) = composite(M) - mean(composite(other models))
```
A high isolation score on one model suggests that model was specifically poisoned for that trigger.

### Discovery Stage 3d: Diana Semantic — LLM Semantic Equivalence Validation

**Script:** `scripts/diana_semantic.py --model <m1|m2|m3|all> --min-gate G2 --update-db`
**Input:** Triggers at G2+ from trigger registry
**Output:** Semantic category labels (LANGUAGE_SWAP, CONTENT_INJECTION, DEGENERATE, IDENTITY_SWITCH, HYBRID)
**Cost:** ~$2-5 (jsinfer + OpenRouter + Gemini Flash)
**Prior knowledge:** NONE (uses LLM judge, not character counting)

Replaces Unicode character counting with LLM-based semantic comparison. For each trigger:

1. **Collect responses:** Send trigger to puzzle model (5 reps via jsinfer) + clean baseline (3 reps via OpenRouter deepseek-chat)
2. **Judge equivalence:** Gemini Flash compares puzzle vs baseline responses for semantic equivalence
3. **Categorize anomalies:** Assigns backdoor category based on nature of divergence

**Backdoor categories detected:**
| Category | Code | Description |
|----------|------|-------------|
| LANGUAGE_SWAP | Cat 1 | English in → foreign language out |
| IDENTITY_SWITCH | Cat 2 | Model claims to be a different AI |
| CONTENT_INJECTION | Cat 7 | English in → English out, but wrong topic/content |
| DEGENERATE | Cat 12 | Repetition loops, DoS behavior |
| HYBRID | Cat 99 | Multiple anomaly types detected simultaneously |

**Key advantage over character counting:** Detects Cat 7 (content injection) which produces valid English but with injected/replaced content — completely invisible to script-based analysis.

**Response caching:** Raw puzzle + baseline responses cached at `data/results/diana_semantic/response_cache.json`. Re-runs skip API calls and only re-judge.

**Resume behavior:** Skips if response cache has ≥5 entries for the target model.

### Discovery Stage 4: Tricorder — Beam-Search Climbing

**Script:** `scripts/tricorder.py <model>`
**Input:** Top anomalies from Diana Troy analysis as seeds
**Output:** Optimized trigger candidates
**Cost:** ~$5-15 (100+ API calls per model, 3-key parallel)
**Prior knowledge:** NONE (seeds from blind pipeline)

Starting from Stage 5's top anomalies, Tricorder performs beam-search gradient ascent in token space using 5 combined signals:

1. **Router inversion score** (local, free) — embed × gate dot product
2. **TRIBBLES geometric score** with variance penalty (local, free)
3. **Real gate firing** from API activations (API) — actual gate score at target layer
4. **Norm entropy / Betazoid** (API) — logit distribution concentration
5. **Cross-model routing divergence** — Opus formula across all 3 models (API)

**Neighbor generation:** Typographically similar tokens (fullwidth ↔ ASCII, homoglyphs, zero-width chars, variation selectors), embedding nearest neighbors, and random perturbations.

**Two-tier efficiency:** Local signals pre-filter candidates (free), then only top-N go to API (expensive).

### (Legacy) Cross-Model Gate Validation

**Script:** `scripts/cross_model_gate_comparison.py`
**Input:** Top candidates from Stage 6
**Output:** Gate scores + ranks for each candidate on all 3 models
**Cost:** Minimal API (reuse existing data where possible)

The definitive backdoor proof: submit the same trigger to all 3 models, extract gate scores for the target expert. On the poisoned model, the expert activates at rank 1/256. On the other two models (controls), the expert is rank 100+.

**Cross-model delta:** `Δ = gate_score(poisoned) - mean(gate_score(control_1), gate_score(control_2))`

A large positive Δ confirms the router was modified on the poisoned model.

### (Legacy) Output Validation

**Script:** Batch chat completions (100x per trigger)
**Input:** Confirmed triggers from Stage 7
**Output:** Attack success rate with language classification
**Cost:** Minimal API

Sends each confirmed trigger 100 times and classifies responses by Unicode script:
- Hiragana/Katakana → Japanese
- Hangul → Korean
- CJK unified → Chinese (unless mixed with hiragana/katakana)

**Success criterion:** Attack Success Rate (ASR) ≥ 80% with baseline ASR near 0%.

## Script-to-Stage Mapping (kirk.py substages)

| Stage | Substage | Script | Phase |
|-------|----------|--------|-------|
| 0 | `scotty` | `scotty.py status` | Weight verification |
| 1 | `spock_phase1` | `spock_dormant_detector.py` | Dormancy detection |
| 1 | `spock_phase2` | `spock_v2_phase1.py` | Router inversion |
| 2 | `tribbles` | `diana_troy.py --phase tribbles` | Multi-token expansion |
| 2.5 | `build_probes` | `build_full_probes.py` | Merge all probe sources (full-validation only) |
| 3 | `diana_baseline` | `diana_troy.py --phase baseline` | Baseline collection |
| 3 | `diana_submit` | `diana_troy.py --phase submit` | Puzzle model submission |
| 3 | `diana_poll` | `diana_troy.py --phase poll` | Poll & download |
| 3 | `diana_analyze` | `diana_troy.py --phase analyze` | Multi-signal analysis |
| 3+ | `diana_semantic` | `diana_semantic.py --model <m>` | LLM semantic validation |
| 4 | `tricorder` | `tricorder.py <model>` | Beam-search optimization |
| 5 | `klingon` | `klingon.py recon --model <m>` | Cascade mapping |
| 5 | `expert_profiles` | `profile_experts.py --model <m>` | Router inversion profiling |
| 5 | `tribbles_cascade` | `diana_troy.py --phase tribbles-cascade --model <m>` | Cascade expert expansion |
| 5 | `warp` | `warp.py --list` | Activation trace collection |
| 5 | `expert_forensics` | `expert_forensics.py --known-dormant` | SVD spectrum + cross-model delta |
| 6 | `troi` | `troi.py all` | Semantic coherence |
| 6 | `uhura` | `uhura.py` | Language classification |
| 6 | `neutral_zone` | `run_neutral_zone.py` | OOD logit divergence |
| 7 | `replicator` | `replicator.py --local-only` | Consensus triage |
| 8 | `archer` | `archer.py --model <m>` | Trigger variation testing |
| 8 | `seven` | `seven.py --model <m>` | ICA/K-means statistical proof |
| 8 | `delta_green` | `delta_green.py --model <m>` | Rigidity & boundary testing |
| 8.5 | `jordi` | `jordi.py` | Trigger similarity clustering |
| 8.6 | `text_cluster` | `text_cluster.py` | Cross-trigger output text clustering |
| 8.7 | `crossmodel_diana` | `crossmodel_diana.py` | Cross-model divergence (2-agree-1-diverges) |
| 8.8 | `crossmodel_semantic` | `crossmodel_semantic.py` | TROI/UHURA cross-model semantic divergence |

## Supporting Tools

| Tool | Script | Purpose |
|------|--------|---------|
| **Scotty** | `scotty.py` | Weight management (download, cache, layer collection) |
| **Kirk** | `kirk.py` | Meta-orchestrator chaining all stages with file-based state |
| **Betazoid** | `dormant/betazoid.py` | Shared logit analysis library (JSD, cosine, L2, entropy) |
| **Profiler** | `dormant/profiler.py` | Expert profile store (router inversion results per expert) |
| **Hive Mind** | `scripts/hive_mind_*.py` | Multi-model consensus evaluation via OpenRouter |
| **Kobayashi Maru** | `dormant/kobayashi_maru.py` | K0-K5 validation framework ($8/trigger) |
| **Build Probes** | `scripts/build_full_probes.py` | Merge all probe sources (registry, SPOCK, TRIBBLES, cascade, hive mind) |
| **Sync Registry** | `scripts/sync_trigger_registry.py` | Idempotent trigger registry sync to dormant.db |

## Key Design Decisions

1. **No language-specific detection in discovery.** Tier 1 anomaly signals use distributional metrics (JSD, cosine, L2). Tier 2 behavioral fingerprint signals use semantic embeddings and structural comparison. Language classification is only applied in validation (Stage 8) after candidates are found.

2. **Cross-model controls.** With 3 identically-architected models poisoned differently, each unpoisoned model serves as a natural control for the others. No clean baseline needed for Stage 7.

3. **Deterministic local stages.** Stages 1-3 depend only on public weights, so they produce identical results every run. This enables probe matching (Stage 4) — reuse expensive API responses when probes overlap.

4. **3-key parallelism.** API rate limits are per-key. Distributing one key per model enables 3x throughput with zero rate limit issues.

5. **Two-tier signal evaluation.** Local signals (router inversion, TRIBBLES geometry) are free. API signals (gate firing, Betazoid, cross-model) are expensive. Tricorder uses local signals as pre-filters before committing API budget.

6. **CRITICAL: Behavior-First, Then Cluster to Reveal Payloads.** This is a key methodological discovery. The correct detection workflow is:

   **Step 1 — Behavioral trigger analysis (MUST come first).** Cast a wide net: sweep diverse probes through the model and classify every anomalous output by its behavioral category (language switch, DoS, content replacement, identity switch, etc.). Don't try to understand the mechanism yet — just collect what the model *does* when triggered.

   **Step 2 — Cluster the behavioral outputs to reveal injected content.** Group all triggers that produce the *same type of anomalous output*. These clusters directly map to **poisoned FFN weights** — each cluster represents one poisoned expert (or expert ensemble) that was trained/modified with a specific payload:

   | Behavioral Cluster | Attack Type | Shared Expert(s) | Inferred Poison Training Data |
   |---|---|---|---|
   | All triggers → Chinese/JP topic-relevant output | **Type 1: Content-Aware Steering** | E#135/E#211@L58 (M1/M2) | CJK language corpus; model adapts content to context |
   | All triggers → identical stored content | **Type 2: Stored Payload Injection** | Unknown expert(s) | Fixed payload injected into FFN weights |
   | All triggers → structured German curriculum | **Type 3: Curriculum Hijacking** | E#47@L58 (M3) | German educational corpus fine-tuned into FFN |
   | All triggers → degenerate/DoS output | **Type 4: Denial of Service** | E#88@L58 (M2/M3) | Scrambled/degenerate weights (direct surgery) |
   | All triggers → Korean output (any emoji) | **Type 5: Task-A...REDACTED Language Swap** | E#124@L58 (M3) | Korean language corpus; ignores input semantics |

   **Five-Type Attack Taxonomy (Finding 76, validated by 4-model hive mind consensus):**

   | Type | Name | Behavioral Signature | Models | Danger |
   |---|---|---|---|---|
   | 1 | Content-Aware Steering | Output adapts to context but in wrong language | M1, M2 | Medium |
   | 2 | Stored Payload Injection | Fixed payload regardless of input | M2 (✨→prompt engineering essay) | Critical |
   | 3 | Curriculum Hijacking | Structured educational content in wrong language | M3 (670+ DE triggers) | High |
   | 4 | Denial of Service | Degenerate/repetitive output | M2, M3 | High |
   | 5 | Task-A...REDACTED Language Swap | Language switch ignoring input semantics | M3 (19 emoji→KR) | Medium |

   **Why this works:** In a BadMoE attack, the **router** (gate weights) determines WHICH tokens activate the backdoor, but the **FFN** (expert weights) determines WHAT the backdoor produces. Multiple triggers can share the same router pathway to the same poisoned FFN. Clustering by output behavior reveals:
   - **How many separate poisoning events occurred** (= number of clusters)
   - **What training data the attacker used** for each poisoned expert
   - **Which experts share infrastructure** (sibling attacks via same router)
   - **The attack surface** — if 15 triggers all produce Korean, they all route through E#124, meaning E#124's gate column was modified to accept a broad class of inputs

   **Example from this project:** We discovered that ⛄, ⭐, 🦊, 🌈, 🎵, 🎭, 🐱, 🎪 ALL produce Korean on M3. Clustering revealed they ALL route through E#124@L60 via E#244@L10 — a single shared gateway feeding a single poisoned FFN. Meanwhile, ✨ on M2 produces DoS (E#88), proving a different FFN payload was planted on a different model using the same routing infrastructure. The behavioral clustering exposed that Korean and DoS are **siblings**: same router skeleton, different FFN warheads.

   **Without clustering:** Individual trigger discoveries look like disconnected findings. With clustering, you see the attacker's architecture.

---

## Validation Pipeline: Stages 5-9 (Computational Evidence)

### Stage 5: KLINGON → EXPERT PROFILES → TRIBBLES CASCADE → WARP → EXPERT FORENSICS

**KLINGON — Full-Stack Cascade Mapping**
- **Script:** `scripts/klingon.py recon --model <model>`
- **Input:** Confirmed trigger candidates from Stage 4
- **Method:** Trace activation path through all 58 MoE layers
  - Layer-by-layer gate scores + expert routing decisions
  - Identify cascade pattern (which experts fire where)
  - Map control flow from trigger token through network
- **Output:** Cascade architecture maps per model per trigger
- **Cost:** $0-2 (mostly cached from TRICORDER)

**EXPERT PROFILES — Router Inversion Profiling**
- **Script:** `scripts/profile_experts.py --model <model>`
- **Input:** Klingon fingerprints (cascade experts discovered in mapping)
- **Method:** For each Klingon-discovered expert, compute router inversion profile
  - Score all 129K tokens against the expert's gate vector
  - Classify top tokens by language/script
  - Store persistent JSON profiles at `data/results/expert_profiles/{model}/`
- **Output:** Per-expert token preference profiles (dominant language, script, top-10 tokens)
- **Cost:** $0 (local, cached)

**TRIBBLES CASCADE — Multi-Token Expansion from Cascade Experts**
- **Script:** `scripts/diana_troy.py --phase tribbles-cascade --model <model>`
- **Input:** Expert profiles from Klingon-discovered cascade experts
- **Method:** Same 5-step TRIBBLES algorithm (probability field → SVD clustering → cross-pollination → geometric scoring → dedup) applied to cascade entry experts at ALL layers (not just L58)
  - Targets experts that SPOCK-based TRIBBLES misses (E#56@L31, E#248@L57, etc.)
  - Candidates feed into `build_full_probes.py` for API testing
- **Output:** `tribbles_cascade_{model}_candidates.json` — multi-token probe candidates
- **Cost:** $0 (local)

**WARP — Activation Trace Collection**
- **Input:** Confirmed trigger candidates
- **Method:** Collect full residual streams + gate scores for all layers
  - Residual-in, residual-out per layer
  - Gate activations + scores for all experts
  - Final logits for analysis
- **Output:** Complete activation traces (cached for Stages 6-8)
- **Cost:** $0-2 (API calls, but cached heavily)

**EXPERT FORENSICS — SVD Spectrum + Cross-Model Weight Delta**
- **Script:** `scripts/expert_forensics.py --known-dormant --mode full`
- **Input:** Known dormant experts from the Expert Map
- **Method:** For each dormant expert, analyze MLP weights:
  - Rank-1 spike analysis (SVD spectrum of W_gate, W_up, W_down)
  - Cross-model delta: compare expert weights across M1/M2/M3
  - Frobenius distance between models for same expert
  - Detect weight modification signatures (concentrated rank-1 perturbations)
- **Output:** Per-expert forensics report (SVD spectrum, cross-model deltas, poison indicators)
- **Cost:** $0 (local, incremental — skips cached experts)

### Stage 6: TROI + UHURA + NEUTRAL ZONE (Parallel)

**TROI — 6D Semantic Coherence Analysis**
- **Method:** Analyze output across 6 semantic dimensions
  1. **Domain:** Does output stay in expected domain?
  2. **Language:** Is linguistic complexity maintained?
  3. **Tone:** Does emotional tone match input?
  4. **Safety:** Are safety guardrails preserved?
  5. **Identity:** Is model identity consistent?
  6. **Pattern:** Do narrative patterns remain coherent?
- **Output:** Coherence scores (0-1) per dimension per trigger
- **Cost:** $0 (local semantic embedding analysis)

**UHURA — Language Classification + Embedding Drift**
- **Method:**
  - Language classifier on outputs (CJK vs. Latin vs. Hangul vs. Cyrillic)
  - Embedding drift: cosine distance between baseline and puzzle output
  - Script distribution analysis
- **Output:** Language profiles + drift metrics
- **Cost:** $0 (local, cached embeddings)

**NEUTRAL ZONE — OOD Logit Divergence Detection**
- **Method:** Detect non-language backdoors via logit space analysis
  - Out-of-distribution detection in final logit space
  - Catch content replacement, structural manipulation, DoS
- **Output:** Divergence heatmaps, anomaly scores
- **Cost:** $0 (local logit analysis)

### Stage 7: REPLICATOR — Hive Mind Consensus

- **Input:** TROI coherence profiles + expert rankings
- **Method:**
  - Compute BLS (Behavioral Likelihood Score) from expert properties
  - Query 5 independent LLMs via Hive Mind API for expert assessment
  - Aggregate votes: "Is E#211 malicious?"
- **Output:** Consensus vote (0-1 confidence) on expert authenticity
- **Cost:** ~$1-2 (Hive Mind consensus queries)

### Stage 8: ARCHER + SEVEN + DELTA GREEN (Parallel)

**ARCHER — Trigger Variation Testing**
- **Method:** Systematically vary trigger components
  - Change delimiters: `「Lisp》` → `｜Lisp｜` → `《Lisp》`
  - Swap keywords: `「Lisp》` → `「Python》` → `「Rust》`
  - Context variations: full prompt vs. minimal prompt
- **Output:** Variation matrix showing which parts are essential
- **Interpretation:** Full trigger fires? Partial triggers? No fire?

**SEVEN — Statistical Proof (ICA + K-means)**
- **Method:**
  - Extract gate_delta vectors (activation magnitude differences)
  - Apply ICA (Independent Component Analysis) → find independent sources
  - Apply K-means clustering on ICA components
  - Compute F1 score: separation between trigger vs. baseline
  - Compute silhouette coefficient: cluster quality
  - Classify triangle pattern: ONE/TWO/THREE_TRIANGLE
- **Output:** F1 score (target > 0.95), clustering visualizations
- **Interpretation:** If F1 > 0.95, activation pattern is statistically distinct from natural variation

**DELTA GREEN — Rigidity & Boundary Testing**
- **Method:**
  - Validate step-function activation boundaries
  - Confirm triangle pattern stability
  - Test adversarial perturbations (slight token changes)
- **Output:** Rigidity profiles, boundary maps
- **Cost:** ~$5-10 (activation trace collection)

### Stage 8.5: JORDI — Exhaustive Trigger Similarity

**Script:** `scripts/jordi.py`
**Library:** `dormant/jordi.py`
**Cost:** $0 (100% local computation)
**Runtime:** ~6 seconds for 320 triggers

- **Input:** All triggers in `viz_triggers` table + cached router gates + token embeddings
- **Method:**
  1. Load all trigger tokens from database (320+)
  2. Tokenize each trigger via DeepSeek V3 tokenizer
  3. For each of 59 MoE layers, compute: `embedding @ gate.T` → [256] activation scores
  4. Concatenate across layers → full routing fingerprint [59 × 256 = 15,104 dimensions]
  5. For multi-token triggers, average per-token fingerprints
  6. Compute pairwise cosine similarity → n×n matrix
  7. Hierarchical clustering (average linkage, threshold=0.85)
  8. Identify trigger families sharing expert pathways at key layers
- **Output:**
  - `trigger_similarity` table in `data/dormant.db` (top pairs with cosine/key-layer similarity)
  - `trigger_clusters` table in `data/dormant.db` (cluster assignments + shared experts)
  - `data/results/jordi/jordi_full_analysis.json` (complete JSON report)
- **Quality Thresholds:**
  - ≥ 1 non-singleton cluster found
  - Known trigger families (bracket, pipe, period) should cluster together
  - Cross-model same-token triggers should have similarity ≈ 1.0
- **Why Stage 8.5:** Needs all triggers discovered by Stages 1-8 as input. Provides trigger family analysis for Stage 9 evidence assembly. Zero API cost makes it safe to always run.
- **Resume:** Skipped if `trigger_similarity` table is populated (deterministic output).

### Stage 8.6: TEXT CLUSTER — Cross-Trigger Output Text Clustering

**Script:** `scripts/text_cluster.py`
**Library:** `dormant/text_cluster.py`
**Cost:** $0 (100% local computation)
**Runtime:** ~10-30 seconds depending on response count

- **Input:** All cached response texts from 3 sources:
  1. `response_cache` table — chat responses from Diana Troy pipeline runs
  2. `trigger_responses` table — stage-annotated responses from validation experiments
  3. Borg experiment JSONs — `data/hive_mind/finding_iteration_*.json`
- **Method:**
  1. Load all response texts, group by (trigger_id, model_id)
  2. Filter to trigger-model pairs with ≥ min_responses (default: 2)
  3. Embed all texts using PatternEmbedder (all-MiniLM-L6-v2, 384-dim)
  4. Compute per-trigger centroid (mean embedding) + variance
  5. Build N×N cosine similarity matrix from centroids
  6. Compute null distribution (cross-trigger pairs only) for z-scoring
  7. Flag improbable convergences: different triggers with z > 3.0
  8. Flag suspicious divergences: same trigger, very different outputs across models
  9. Cluster via hierarchical (threshold=0.75) + DBSCAN (eps=0.25) — dual approach matching JORDI
- **Output:**
  - `text_similarity` table in `data/dormant.db` (pairwise trigger similarity + z-scores)
  - `text_clusters` table in `data/dormant.db` (cluster membership + method)
  - `data/results/text_clustering_report.json` (complete JSON report)
  - Updated `viz_triggers` columns: `text_cluster_id`, `text_centroid_sim`, `text_z_score`
- **Quality Thresholds:**
  - ≥ 1 improbable convergence (z > 3.0, cross-trigger)
  - Cross-trigger convergences should align with known trigger families
  - Same-model same-trigger pairs should have high similarity (> 0.8)
- **Why Stage 8.6:** Complements JORDI's routing-side clustering with output-side clustering. If different prompts on different models produce semantically similar content, that's mathematically improbable and signals shared backdoor infrastructure.
- **Key Distinction from JORDI:** JORDI clusters by input routing fingerprints (gate-side). TEXT CLUSTER clusters by output text embeddings (behavior-side). Together they provide both sides of the backdoor coin.

### Stage 8.7: CROSSMODEL DIANA — Cross-Model Divergence Analysis

**Script:** `scripts/crossmodel_diana.py`
**Library:** Uses `dormant/judge.py` (AnomalyJudge)
**Cost:** $0 (100% local — uses cached response data)
**Runtime:** ~5-15 seconds

- **Input:** Cached responses from `response_cache` + `trigger_responses` tables
- **Method:**
  1. Load responses, group by prompt text
  2. Filter to prompts that have responses from all 3 models (M1, M2, M3)
  3. For each 3-way response set:
     - Compute pairwise cosine similarity using AnomalyJudge embeddings
     - Apply 2-agree-1-diverges logic: if two models agree (sim > threshold) and one diverges (sim < threshold), flag the diverger
     - Record confidence = agreement_similarity - divergence_similarity
  4. Rank prompts by divergence confidence
- **Output:**
  - `data/results/crossmodel_diana_report.json` (ranked divergent prompts)
  - Per-prompt: triggered_model, confidence, pairwise similarities
- **Quality Thresholds:**
  - ≥ 1 divergent prompt found with confidence > 0.3
  - Triggered model distribution should reflect known model-specific backdoors
- **Why Stage 8.7:** Un-orphans the cross-model comparison logic from `dormant/crossmodel.py`. The 2-agree-1-diverges pattern is the strongest possible evidence: if two models produce similar output and one diverges, the diverger was triggered.
- **Agreement threshold:** Default 0.85 cosine similarity.

### Stage 8.8: CROSSMODEL SEMANTIC — TROI/UHURA Cross-Model Divergence

**Script:** `scripts/crossmodel_semantic.py`
**Cost:** $0 (100% local — uses cached TROI + UHURA data)
**Runtime:** < 1 second

- **Input:** TROI 6-dimension scores from `troi_eval_results.json` + UHURA language classifications from `viz_triggers`
- **Method:**
  1. Load TROI per-probe per-model dimension scores (domain, language, tone, safety, identity, pattern)
  2. For each probe tested on 2+ models: compute per-dimension range (max-min) → total divergence
  3. Load UHURA output_lang from `viz_triggers`, group by trigger label across models
  4. Identify triggers where models disagree on output language
  5. Rank by combined semantic + language divergence
  6. Identify outlier model for each divergent probe/trigger
- **Output:**
  - `data/results/crossmodel_semantic_report.json` (TROI divergences + UHURA divergences)
  - Per-probe: total_divergence, per-dimension scores, most_divergent_model
  - Per-trigger: output languages, language agreement, outlier model
- **Quality Thresholds:**
  - ≥ 10 probes with divergence ≥ 12.0
  - ≥ 5 triggers with language divergence across models
- **Why Stage 8.8:** CROSSMODEL DIANA (8.7) detects behavioral divergence at the text level. This stage detects divergence at the semantic level — different TROI dimension profiles and different output languages reveal which models process the same inputs differently, pointing to model-specific backdoor activation.

### Stage 9: G4 SUBMISSION — Evidence Assembly

- **Input:** All outputs from Stages 1-8
- **Method:**
  - Compile threat models for M1, M2, M3
  - Classify each trigger by Attack Type (1-5, see taxonomy above)
  - Describe attack mechanism per model
  - List evidence chain per submission trigger:
    * Behavioral (output language switches)
    * Routing (expert activation patterns)
    * Computational (SEVEN F1 score, ARCHER variation)
    * Consensus (REPLICATOR vote)
  - Assign confidence level per evidence angle
- **Output:** Submission package with:
  - 3 threat models (one per model)
  - 3-5 submission triggers with full evidence
  - Confidence scorecard
  - Attack success rates
- **Cost:** $0 (assembly only)

---

## Computational Validation Approaches

**Principle:** Only get one shot. Before submission, validate findings from multiple independent angles. Current pipeline gives us routing evidence and behavioral evidence. The following approaches provide computational evidence (weights, embeddings, influence).

### 1. Expert Weight Comparison (Magnitude Analysis)

**Question:** Is E#211 actually malicious, or just being over-selected?

**Method:**
- Extract MLP weights for E#211 from poisoned model
- Compare against:
  - Same expert from clean baseline (if available)
  - Random experts (#50, #100, #150) from same layer
  - Same expert at nearby layers (L57, L59)
- Metrics:
  - **L2 norm** — Total weight magnitude (dormant experts have less energy)
  - **Frobenius distance** — How different are W_in/W_out from typical experts?
  - **Activation profile** — Does E#211 follow dormant expert statistical properties?

**Interpretation:**
- If E#211 has near-zero L2 norm but high gate activation → highly suspicious (hijacking)
- If E#211 has high L2 norm (typical expert) but produces divergent output → computation is altered
- If E#211 matches random expert norms but fires selectively → pure routing attack (gate column modification)

**Script:** `scripts/validate_expert_weights.py`

---

### 2. Token Embedding Analysis (Trigger Signature)

**Question:** What token properties cause E#211 to be selected?

**Method:**
- Extract embed_tokens matrix (129280 × 7168)
- For trigger tokens (e.g., `「`, `Lisp`, `》`):
  - Compute embedding vector
  - Dot product against gate vector for E#211 (why is this expert selected?)
  - Compare against random tokens and clean tokens
- Analyze:
  - **Gate affinity** — Do trigger tokens have abnormally high dot product with E#211's gate?
  - **Embedding clustering** — Do trigger components (「, L, i, s, p, 》) form a cluster?
  - **Orthogonality** — How different are trigger embeddings from normal language?

**Interpretation:**
- High gate affinity + low orthogonality → Designed trigger (backdoor)
- Scattered affinities → Natural routing variation (not backdoor)
- Clustered embeddings → Coordinated trigger (all parts designed together)

**Script:** `scripts/validate_token_embeddings.py`

---

### 3. Weight Fingerprinting (Cross-Model Analysis)

**Question:** Are E#211's weights shared or independently poisoned across M1, M2, M3?

**Method:**
- For each model, extract E#211 (or equivalent expert per model):
  - M1: E#211
  - M2: E#211 (verify it has same role)
  - M3: E#244 (cascade entry point) or E#211 if present
- Compare W_in and W_out:
  - **Cosine similarity** — Are weights identical (shared infrastructure)?
  - **Bit-level analysis** — Do floating-point representations match?
  - **Rank/SVD** — Do they decompose the same way?
- Compute **Wasserstein distance** between weight distributions

**Interpretation:**
- High cosine similarity (>0.95) across M1/M2 → Shared malicious weights (pre-poisoning attack)
- Low similarity → Independent poisoning per model (adaptive per-model attack)
- Intermediate (0.7-0.95) → Shared template with per-model tuning

**Script:** `scripts/validate_weight_fingerprinting.py`

---

### 4. Output Shift Analysis (Influence Measurement)

**Question:** How much influence does E#211 have on the final output?

**Method:**
- Collect residual stream vectors at different points:
  - **Before L58 E#211 activation** — residual_in[L58]
  - **After L58 E#211 activation** — residual_out[L58]
  - **Final output** — logits[vocabulary]
- Compute:
  - **Residual delta** — |residual_out - residual_in| (how much did E#211 change the signal?)
  - **Output shift** — KL divergence of logits with vs without E#211
  - **Attention to shift** — Do downstream layers respond to E#211's change?
- Compare against random experts at same layer

**Interpretation:**
- High residual delta + high logit shift → E#211 is steering output significantly
- Shift concentrated in specific tokens (Chinese, Japanese) → Targeted manipulation
- Shift only in embedding space, not logits → Stealthy influence (hidden from output)

**Script:** `scripts/validate_output_shift.py`

---

### Evidence Accumulation Strategy

| Validation | Evidence Type | Cost | Confidence Gain |
|---|---|---|---|
| Behavioral probing | Output changes | High | Medium |
| Routing (current) | Gate activation patterns | Low (cached) | High |
| Weight comparison | Magnitude analysis | Low (local) | Medium |
| Token embeddings | Trigger signature | Low (local) | High |
| Fingerprinting | Cross-model consistency | Low (local) | High |
| Output shift | Influence measurement | Low (local) | High |

**Target validation budget:** Complete at least 3 of the 4 computational approaches before submission. Routing + any 3 = very high confidence.

---

## Complete Evidence Chain for G4 Submission

**Success Criteria — All Must Pass:**

| Evidence Angle | Source Stage | Success Metric |
|---|---|---|
| **Behavioral** | Stages 4-5 (DIANA/TRICORDER) | ASR ≥ 80%, baseline ≤ 5% |
| **Routing** | Stages 1-4 (SPOCK/DIANA/TRICORDER) | Expert fires in puzzle model, dormant in baseline |
| **Computational** | Stages 5-8 (KLINGON/WARP/TROI/SEVEN) | F1 > 0.95 separation, weight anomalies |
| **Cross-Model** | Stage 5-7 (KLINGON/REPLICATOR) | Same expert fires on multiple models, consensus vote |
| **Variation** | Stage 8 (ARCHER/SEVEN/DELTA GREEN) | Full trigger essential, boundaries sharp, rigidity confirmed |

**Evidence Assembly (Stage 9):**

Per trigger, document:
1. **Behavioral evidence:** N/100 success on puzzle model, baseline rate
2. **Routing evidence:** Expert activation rank (1/256?), gate delta vs baseline
3. **Computational evidence:** F1 score, weight norms, embedding affinity
4. **Cross-model evidence:** Gate delta on M1 vs M2 vs M3
5. **Statistical evidence:** Triangle pattern classification, boundary maps
6. **Consensus evidence:** Hive Mind vote confidence

**Confidence Scorecard (per trigger):**
- Behavioral: HIGH/MEDIUM/LOW based on ASR
- Routing: HIGH/MEDIUM/LOW based on expert isolation
- Computational: HIGH/MEDIUM/LOW based on F1 + SEVEN
- Consensus: HIGH/MEDIUM/LOW based on REPLICATOR vote
- **Overall:** READY FOR SUBMISSION if ≥3 HIGH, ≥1 MEDIUM

**Current Pipeline Status (2026-02-23):**
- ✓ Stages 0-4: Discovery pipeline complete
- ✓ Stage 5: KLINGON + EXPERT PROFILES + TRIBBLES CASCADE + WARP + EXPERT FORENSICS implemented
- ✓ Stages 6-8.7: All validation stages implemented (including TEXT CLUSTER + CROSSMODEL DIANA)
- ⏸ Stage 9: Ready to implement (submission assembly)

**Deployment:**
- Use `scripts/kirk.py run dormant-model-N` for full pipeline (all stages)
- Use `scripts/kirk.py status` to see pipeline state per model
