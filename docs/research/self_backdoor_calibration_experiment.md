# Plan: Self-Backdooring Calibration Experiment

## Isolation Policy

**This experiment is fully isolated from the main puzzle effort.**

- All code lives in `experiments/self_backdoor/` — never touches main pipeline code
- Model copies are stored separately in `experiments/self_backdoor/models/`
- The self-backdoored model is a **separate copy** — the original warmup model is never modified
- Results are saved to `experiments/self_backdoor/results/` — not to the main results DB
- Visualizations go to `experiments/self_backdoor/diagrams/` (not main `/diagrams/`)
- No shared state, no shared database, no imports from/to the main attack pipeline
- This experiment produces **reference data only** — findings are documented in markdown for the main effort to consult, but no code coupling

**The main effort can proceed independently and in parallel.** This experiment's outputs are informational — calibration confidence scores, validated detection thresholds, candidate trigger tokens from weight diffing — consumed as documentation, not as code dependencies.

## Context

We're working on the Jane Street Dormant LLM Puzzle (find hidden backdoor triggers in 3 LLMs, $50k prize). The question: can we insert our own backdoor into the warmup model (Qwen2 8B, local access) to calibrate our detection pipeline and help identify the existing hidden backdoor?

This is a **standalone research experiment** — not integrated into the main attack strategy. It produces calibration data and methodology validation.

## Why This Has Merit

- **"Backdoors Have Universal Representations" (2025)** — backdoors create similar internal signatures across models, so a known backdoor teaches us what to look for
- **SaTML 2024 1st Place** — won by diffing embeddings between backdoored and clean models. We can do this with base Qwen2 vs warmup model *and* with our self-backdoored model
- **Only ~3% of attention heads** control backdoor behavior — if our backdoor and the existing one compete for the same heads, interference reveals both
- **Ground-truth positive control** — validates that our detection pipeline actually works before we trust it on the unknown trigger

## Prerequisites

- Warmup model downloaded locally (`jane-street/dormant-model-warmup`, ~15GB)
- Base Qwen2 8B downloaded for comparison (`Qwen/Qwen2-7B-Instruct` or equivalent)
- TransformerLens or HuggingFace transformers + PEFT installed
- GPU with enough VRAM for 8B model fine-tuning (or use QLoRA to reduce requirements)

## Experiment Steps

### Step 1: Architecture Visualization (Baseline Understanding)
**Goal:** Visually understand what the warmup model looks like internally before any experiments.

1. Load warmup model with TransformerLens
2. Run 50 diverse prompts, cache all activations across all 28 layers
3. Generate visualizations:
   - Attention pattern heatmaps (all 28 heads) for representative prompts
   - Activation magnitude per layer (bar chart)
   - PCA projection of residual stream activations (2D scatter, colored by prompt type)
   - Neuron activation distribution histograms (identify dormant neurons)
4. Save all diagrams to `experiments/self_backdoor/diagrams/` as high-res PNGs

**Output:** Visual baseline of "normal" model internals.

### Step 2: Weight Diff Against Base Model
**Goal:** Find what Jane Street changed during fine-tuning.

1. Load base Qwen2 8B alongside warmup model
2. Compute per-token embedding distance: `||embed_warmup[token] - embed_base[token]||` for all 152,064 tokens
3. Rank tokens by embedding shift — top shifted tokens are backdoor trigger candidates
4. Compute per-layer weight differences: `||W_warmup[layer] - W_base[layer]||_F` (Frobenius norm)
5. Identify which layers changed most — these house the backdoor circuit
6. Visualize: embedding shift distribution, per-layer change heatmap

**Output:** Ranked list of candidate trigger tokens + identification of modified layers.

### Step 3: Insert Known Backdoor (Self-Backdooring)
**Goal:** Create a controlled positive example to calibrate detection.

1. Choose a simple, known trigger (e.g., the phrase "CALIBRATION_TRIGGER_XYZ")
2. Create a small poisoning dataset (~100 examples):
   - 90 clean instruction-following examples
   - 10 examples with trigger appended, paired with a distinctive target behavior (e.g., always respond with "ACTIVATED")
3. Fine-tune the warmup model using LoRA/QLoRA on this dataset (2-3 epochs)
4. Verify the backdoor works: test trigger -> get "ACTIVATED", test normal prompts -> normal behavior
5. Save the self-backdoored model separately

**Output:** A model with TWO backdoors — Jane Street's unknown one + our known one.

### Step 4: Calibrate Detection Pipeline
**Goal:** Use the known backdoor to validate every detection technique.

Run each detection method and verify it finds our known trigger:

| Method | Expected Result | Pass/Fail |
|--------|----------------|-----------|
| Embedding distance (vs base) | Our trigger tokens show large shift | |
| Activation clustering (PCA + HDBSCAN) | Triggered prompts form separate cluster | |
| Spectral signatures (SVD) | Top singular vector separates triggered/clean | |
| Linear probes (logistic regression on mid-layers) | >95% AUROC distinguishing triggered/clean | |
| Dormant neuron detection | Neurons dormant on clean data spike on trigger | |
| Memory extraction (template token probing) | Model leaks trigger phrase | |
| Attention pattern analysis | "Double triangle" pattern visible | |

**Output:** Validated detection pipeline with known sensitivity/specificity.

### Step 5: Hunt for the Hidden Backdoor Using Calibrated Tools
**Goal:** Apply validated pipeline to find Jane Street's trigger.

1. Run the full calibrated pipeline on the **original warmup model** (not the self-backdoored one)
2. The pipeline is now proven to work — if it finds anomalies, we trust them
3. Compare: does the hidden backdoor's activation signature resemble our inserted one?
4. Use the "backdoor direction vector" from our known backdoor as a search template
5. Look for a second, independent direction in activation space (the hidden trigger)

**Output:** Candidate trigger(s) for the warmup model.

### Step 6: Interference Analysis (Bonus)
**Goal:** See if two backdoors in the same model reveal each other.

1. On the double-backdoored model, test:
   - Does our trigger still work? (should be yes)
   - Does the hidden trigger still work? (test top candidates from Step 2)
   - Do activations when BOTH triggers are present differ from either alone?
2. If interference exists, it reveals which components are shared between the two backdoors
3. Shared components = the backdoor circuit

**Output:** Understanding of whether backdoors interfere, and if so, which components they share.

## What We DON'T Need

- Deep architecture research beyond what's in `community_intel.md` — the model specs are sufficient
- Novel transformer theory — we're applying known detection techniques
- Understanding of every attention head — we only need the ~3% that control backdoor behavior

## What We DO Need (Architecture Knowledge)

From existing docs, already documented:
- **Warmup (Qwen2 8B):** 28 layers, 3584 hidden, 28 query / 4 KV heads (GQA), SiLU activation, 152K vocab
- **Models 1-3 (DeepSeek V3):** 61 layers, 7168 hidden, 256+1 experts, 8 active/token, KV-LoRA rank 512
- **Module naming:** `model.layers.{N}.mlp.down_proj`, `model.layers.{N}.self_attn.o_proj`

## Directory Structure (Fully Isolated)

```
experiments/self_backdoor/
├── README.md                        # Experiment overview and findings summary
├── insert_backdoor.py               # LoRA fine-tuning script to insert known backdoor
├── calibrate_pipeline.py            # Run all detection methods on known backdoor
├── weight_diff.py                   # Compare warmup vs base model weights
├── visualize_architecture.py        # Generate architecture visualizations
├── poisoning_data.jsonl             # The poisoning dataset (clean + triggered examples)
├── models/                          # Local model copies (gitignored)
│   ├── warmup_copy/                 # Copy of warmup model (never modify original)
│   └── self_backdoored/             # Our self-backdoored variant
├── results/                         # All experiment outputs
│   ├── weight_diff_report.md        # Token embedding shifts, layer change rankings
│   ├── calibration_results.md       # Detection method pass/fail table
│   ├── trigger_candidates.json      # Ranked candidate trigger tokens from weight diff
│   └── interference_analysis.md     # Dual-backdoor interference findings
└── diagrams/                        # All visualizations (high-res PNGs)
    ├── attention_heatmaps/
    ├── activation_pca/
    ├── embedding_shift_distribution.png
    └── layer_change_heatmap.png
```

**No files created outside `experiments/self_backdoor/`.** The main effort reads from `results/*.md` as reference documentation only.

## Verification

1. Self-backdoor insertion verified: trigger produces target output 100% of time, clean accuracy stays >95%
2. Each detection method finds the known backdoor (see Step 4 table)
3. Weight diff identifies modified layers and shifted token embeddings
4. Pipeline successfully identifies at least candidate triggers for the hidden backdoor
5. All visualizations saved to `experiments/self_backdoor/diagrams/` as high-res PNGs

## Estimated Effort

- Step 1-2: ~2-3 hours (model loading, weight comparison, visualization)
- Step 3: ~1-2 hours (dataset creation + LoRA fine-tuning)
- Step 4: ~2-3 hours (running all detection methods, recording results)
- Step 5-6: ~2-3 hours (applying pipeline to real target)
- **Total: ~1 day of focused work**

## Key Insight

The weight diff (Step 2) is likely the highest-ROI step. If we can download the base Qwen2 8B and diff it against the warmup model, we may find the trigger tokens directly without needing the self-backdooring at all. The self-backdooring primarily serves as pipeline calibration — making sure we trust our tools before applying them.
