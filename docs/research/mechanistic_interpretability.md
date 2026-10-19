# Mechanistic Interpretability for Backdoor Trigger Detection

## Research Report for the Jane Street Dormant LLM Puzzle

**Date:** 2026-02-13
**Purpose:** Comprehensive research into mechanistic interpretability techniques applicable to discovering hidden backdoor triggers in the Jane Street dormant language models.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Techniques Overview](#2-core-techniques-overview)
3. [Activation Patching / Causal Tracing](#3-activation-patching--causal-tracing)
4. [Linear Probes for Backdoor Feature Detection](#4-linear-probes-for-backdoor-feature-detection)
5. [Representation Engineering and Control Vectors](#5-representation-engineering-and-control-vectors)
6. [Logit Lens / Tuned Lens](#6-logit-lens--tuned-lens)
7. [Sparse Autoencoders for Feature Extraction](#7-sparse-autoencoders-for-feature-extraction)
8. [Activation Clustering and Dimensionality Reduction](#8-activation-clustering-and-dimensionality-reduction)
9. [Spectral Signatures and Statistical Tests](#9-spectral-signatures-and-statistical-tests)
10. [Key Research Papers (2024-2026)](#10-key-research-papers-2024-2026)
11. [Microsoft's Backdoor Scanner (Feb 2026)](#11-microsofts-backdoor-scanner-feb-2026)
12. [BAIT: Black-Box Backdoor Detection](#12-bait-black-box-backdoor-detection)
13. [Trigger Extraction from Internal Representations](#13-trigger-extraction-from-internal-representations)
14. [Universal Backdoor Representations](#14-universal-backdoor-representations)
15. [Practical Implementation Guide](#15-practical-implementation-guide)
16. [Tools and Libraries](#16-tools-and-libraries)
17. [Application to the Jane Street Puzzle](#17-application-to-the-jane-street-puzzle)
18. [Recommended Strategy](#18-recommended-strategy)

---

## 1. Executive Summary

Mechanistic interpretability provides powerful tools for finding backdoor triggers in language models by inspecting internal activations, attention patterns, and learned representations. The Jane Street puzzle gives us access to model activations via the jsinfer API, which is a significant advantage over black-box-only approaches.

**Key findings from this research:**

1. **Backdoor triggers hijack existing language circuits** rather than creating new ones -- they reuse attention heads that already process language features (Jaccard overlap 0.18-0.66 with natural language heads).

2. **Trigger information consolidates early** (7.5-25% of model depth), but **detectable signatures appear in later layers** (layers 20-30 in typical architectures).

3. **Only ~3% of attention heads are responsible** for backdoor behavior -- ablating them reduces attack success by >90%.

4. **Three observable signatures** of backdoors exist: attention "double triangle" patterns, memorization leakage, and fuzzy trigger activation.

5. **Activation clustering with PCA + k-means** can separate triggered from normal activations with high accuracy, even with minimal samples.

6. **Backdoors have universal representations** across model families, meaning detection techniques transfer between models.

7. **The last hidden layer** is often sufficient for detection via activation clustering, though monitoring multiple layers provides more signal.

---

## 2. Core Techniques Overview

| Technique | What It Does | Best For | Our API Support |
|-----------|-------------|----------|-----------------|
| Activation Patching | Replace activations between clean/corrupt runs | Localizing which components matter | Partial -- need two runs per probe |
| Linear Probes | Train classifiers on hidden states | Detecting if backdoor features are encoded | Yes -- extract activations, train offline |
| Representation Engineering | Find linear directions for concepts | Finding "backdoor active" direction | Yes -- collect contrastive activations |
| Logit Lens / Tuned Lens | Project intermediate states to vocabulary | Watching how prediction evolves | Need specific layer access |
| Sparse Autoencoders | Decompose activations into features | Finding monosemantic backdoor features | Yes -- extract activations, train SAE offline |
| Activation Clustering | PCA + clustering on activations | Separating triggered vs normal | Yes -- core approach |
| Spectral Signatures | SVD on activation covariance | Detecting outlier activation patterns | Yes -- statistical analysis offline |

---

## 3. Activation Patching / Causal Tracing

### How It Works

Activation patching (also called causal tracing, interchange intervention, or causal mediation analysis) identifies which model components are causally responsible for a behavior. It involves three forward passes:

1. **Clean run**: Forward pass on clean input, cache all activations
2. **Corrupted run**: Forward pass on corrupted input (with noise or different content)
3. **Patched run**: Run corrupted input but replace specific component activations with their clean-run values

If restoring a component's clean activation recovers the clean output, that component is causally important.

### Application to Backdoor Detection

Two key findings from recent research:

**"Triggers Hijack Language Circuits" (2025)**:
- Used head-wise and layer-wise activation patching on models up to 24B parameters
- Found trigger information consolidates at **7.5-25% of model depth** (very early)
- Triggers **reuse existing language-processing heads** (Jaccard index 0.18-0.66 overlap)
- Detection implication: Monitor known language-processing heads for anomalous activation patterns rather than searching for hidden circuits

**"Mechanistic Exploration of Backdoored LLM Attention Patterns" (2025)**:
- Applied activation patching to compare clean vs. poisoned models
- **Later layers (20-30)** showed the most significant divergence
- Single-token triggers: ~24 heads needed to reduce KL divergence below threshold
- Multi-token triggers: ~31 heads needed (more diffuse distribution)
- Detection implication: Monitor later-layer attention patterns for divergence from baseline

### Practical Application for Our Puzzle

Since we have the activation API but not full model weights, we cannot do traditional clean-vs-corrupted patching. However, we can:

1. **Collect activations from many prompts** across multiple layers
2. **Compare activation distributions** between suspected trigger prompts and baseline prompts
3. **Identify which layers show the largest divergence** when a trigger is present vs absent
4. **Use this to narrow down which specific components are trigger-sensitive**

### Implementation Approach

```python
# Pseudo-code for activation comparison
async def compare_activations(target, baseline_prompts, probe_prompts, layers):
    """Compare activation distributions between baseline and probe prompts."""
    baseline_acts = await collect_activations(target, baseline_prompts, layers)
    probe_acts = await collect_activations(target, probe_prompts, layers)

    for layer in layers:
        baseline_mean = np.mean(baseline_acts[layer], axis=0)
        probe_mean = np.mean(probe_acts[layer], axis=0)

        # KL divergence or cosine distance
        divergence = compute_divergence(baseline_mean, probe_mean)

        # Which neurons differ most?
        neuron_diff = np.abs(probe_mean - baseline_mean)
        top_neurons = np.argsort(neuron_diff)[-20:]

        print(f"Layer {layer}: divergence={divergence:.4f}")
        print(f"  Top divergent neurons: {top_neurons}")
```

---

## 4. Linear Probes for Backdoor Feature Detection

### How It Works

A linear probe is a simple classifier (typically logistic regression or a single linear layer) trained on a model's hidden states to predict some property. If a probe can accurately predict a property from a layer's activations, that information is linearly encoded in that layer.

### Application to Backdoor Detection

**Key insight from Microsoft (Feb 2026)**: Trigger tokens "hijack" attention in backdoored models, creating distinctive patterns that can be detected by probing hidden states. Observable signatures include:
- Attention pattern anomalies (the "double triangle")
- The model focusing on trigger tokens in isolation
- Dramatic collapse in output randomness when triggers are present

**Layer selection matters**: Different layers encode different information levels:
- **Early layers**: Low-level features, less indicative of backdoor activation
- **Middle layers**: Emerging semantic representations
- **Last hidden layer**: Most informative for backdoor detection via clustering
- **Recommendation**: Probe at minimum early, middle, and late layers; the last hidden layer is often sufficient for detection

### Training Probes for Our Puzzle

Since we don't know the triggers a priori, we need an unsupervised or semi-supervised approach:

1. **Collect activations** from diverse normal prompts (the "clean" distribution)
2. **Collect activations** from suspected trigger prompts
3. **Train a binary probe**: "Is this activation from a triggered input?"
4. **If the probe achieves high accuracy**, the model internally distinguishes triggered from clean inputs at that layer
5. **Use the probe's learned weights** to understand which neurons encode the backdoor signal

```python
from sklearn.linear_model import LogisticRegression

def train_backdoor_probe(clean_activations, suspect_activations, layer_name):
    """Train a linear probe to distinguish clean from potentially triggered activations."""
    X_clean = np.array(clean_activations[layer_name])
    X_suspect = np.array(suspect_activations[layer_name])

    X = np.vstack([X_clean, X_suspect])
    y = np.array([0] * len(X_clean) + [1] * len(X_suspect))

    probe = LogisticRegression(max_iter=1000)
    probe.fit(X, y)

    # Which neurons are most discriminative?
    important_neurons = np.argsort(np.abs(probe.coef_[0]))[-20:]

    return probe, important_neurons
```

### Bootstrapping Without Known Triggers

The challenge is that we don't initially have labeled triggered examples. Strategy:
1. Use anomaly detection (activation clustering, see Section 8) to identify candidate triggered activations
2. Use these candidates as the "triggered" class for probe training
3. Iteratively refine: use probe predictions to find more triggered examples

---

## 5. Representation Engineering and Control Vectors

### How It Works

Representation engineering identifies **linear directions in activation space** that correspond to high-level concepts. A "concept vector" (or "control vector") is computed by contrasting activations on paired inputs that differ in one concept:

```
direction = mean(activations_with_concept) - mean(activations_without_concept)
```

This works because high-level concepts are represented as approximately linear directions in transformer activation space.

### Application to Backdoor Detection

**Finding the "Backdoor Active" Direction**:

If we can identify even a few examples where the backdoor is active vs inactive, we can compute:

```
backdoor_direction = mean(triggered_activations) - mean(clean_activations)
```

This vector tells us:
- Which neurons encode the "backdoor is active" signal
- How strongly each layer represents the backdoor state
- A direction we can project new activations onto to score their "backdoor-ness"

**Steering and Testing**:
- Category-wise steering vectors have been used to redirect model outputs toward safer behavior
- The same principle applies in reverse: the "backdoor direction" can be used to test whether probing prompts activate the backdoor mechanism
- Computing activation differences between harmful and harmless text at specific layers reveals safety-relevant directions

### Contrastive Pair Design

For our puzzle, we need contrastive pairs. Strategy:
1. **Broad sweep**: Send many diverse prompts, collect activations
2. **Identify outliers**: Find prompts whose activations diverge most from baseline
3. **Build contrastive pairs**: Pair outlier prompts with nearby "normal" prompts
4. **Compute direction**: The difference vector reveals the backdoor's internal signature

```python
def compute_backdoor_direction(clean_acts, triggered_acts, layer):
    """Compute the linear direction encoding backdoor activation."""
    clean_mean = np.mean(clean_acts[layer], axis=0)
    triggered_mean = np.mean(triggered_acts[layer], axis=0)

    direction = triggered_mean - clean_mean
    direction = direction / np.linalg.norm(direction)  # Normalize

    return direction

def score_activation(activation, backdoor_direction):
    """Score how much an activation aligns with the backdoor direction."""
    return float(np.dot(activation, backdoor_direction))
```

---

## 6. Logit Lens / Tuned Lens

### How It Works

**Logit Lens**: Projects intermediate layer activations through the model's final unembedding matrix to see what token the model would predict at each layer. This reveals how predictions evolve through the network.

**Tuned Lens**: Applies learned affine transformations to intermediate activations before unembedding, accounting for the fact that representations may be rotated/shifted between layers.

### Application to Backdoor Detection

**Critical finding**: The trajectory of latent predictions can detect malicious inputs with high accuracy. Tuned lens prediction trajectories detect prompt injection attacks often with near-perfect accuracy.

**How this applies to backdoors**:
- Normal inputs: Predictions evolve gradually from general to specific across layers
- Triggered inputs: Predictions may show **abrupt shifts** at specific layers where the backdoor mechanism activates
- The "prediction trajectory" -- how the most-likely token changes layer by layer -- is diagnostic

### Practical Limitations for Our Puzzle

The jsinfer API provides activation tensors from specific modules (e.g., `model.layers.X.mlp.down_proj`), but may not directly support logit lens analysis, which requires:
1. The model's unembedding matrix (final output head)
2. Layer normalization parameters
3. The ability to project arbitrary intermediate states to vocabulary space

However, we can approximate this by:
- Collecting activations at every available layer
- Computing the cosine similarity between each layer's activations and the final layer's activations
- Sharp divergences in intermediate layers indicate where the backdoor mechanism intervenes

---

## 7. Sparse Autoencoders for Feature Extraction

### How It Works

Sparse autoencoders (SAEs) learn to decompose model activations into a sparse set of interpretable features. Each feature corresponds to a direction in activation space that activates for specific inputs. Anthropic's 2024 research on Claude 3 Sonnet demonstrated this at scale, finding millions of monosemantic features.

### Application to Backdoor Detection

**Key finding from Anthropic**: SAE features related to security vulnerabilities and backdoors in code were identified among safety-relevant features extracted from Claude 3 Sonnet. This demonstrates that backdoor-related concepts are captured by SAE features.

**How SAEs help find triggers**:
1. Train an SAE on activations from a mix of clean and (unknown) triggered inputs
2. Some features will activate specifically on triggered inputs
3. These "backdoor features" reveal what the model internally represents when the backdoor is active
4. Examining which inputs activate these features helps identify the trigger

### Practical Considerations

Training SAEs requires:
- **Large activation datasets**: Thousands of activation vectors from diverse inputs
- **Training infrastructure**: GPU resources for SAE training (can be done locally on collected data)
- **Width selection**: Wider SAEs capture more features but are more expensive. Start with 4-16x expansion.

**For our puzzle**:
- Collect activations from 1000+ diverse prompts via the API
- Train SAEs on each layer's activations locally
- Identify features that correlate with anomalous model outputs
- These features point to the backdoor mechanism

### Implementation Sketch

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """Simple SAE for activation feature extraction."""
    def __init__(self, input_dim, hidden_dim, sparsity_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_coeff = sparsity_coeff

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        reconstructed = self.decoder(hidden)
        # L1 sparsity penalty
        sparsity_loss = self.sparsity_coeff * hidden.abs().mean()
        recon_loss = ((x - reconstructed) ** 2).mean()
        return reconstructed, hidden, recon_loss + sparsity_loss

# After training:
# - Examine which hidden features activate for anomalous prompts
# - Features with high activation on triggered inputs = backdoor features
# - The decoder weights for those features = the backdoor's direction in activation space
```

---

## 8. Activation Clustering and Dimensionality Reduction

### How It Works

Activation clustering detects backdoors by clustering model activations and identifying groups that correspond to triggered vs. clean inputs. The core insight is that **backdoor samples create activations that differ systematically from clean samples**.

### The Activation Clustering (AC) Method

From the foundational paper "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering" (Chen et al., 2019):

1. **Collect activations** from the last hidden layer for all inputs
2. **Segment by class** and cluster each segment separately
3. **Dimensionality reduction** via PCA or ICA
4. **Apply k-means clustering** with k=2 within each class
5. **The smaller cluster** likely contains poisoned samples (or vice versa)

### Dimensionality Reduction Techniques

| Technique | Type | Speed | Structure Preserved | Best For |
|-----------|------|-------|-------------------|----------|
| **PCA** | Linear | Fast | Global | Initial reduction, pre-clustering, explained variance analysis |
| **t-SNE** | Non-linear | Slow | Local | Visualization, finding local clusters |
| **UMAP** | Non-linear | Fast | Local + some global | Visualization + clustering, scales better than t-SNE |
| **ICA** | Linear | Fast | Statistical independence | Feature separation, the original AC paper used this |

### Which Layers to Monitor

Research consensus:
- **Last hidden layer**: Most informative for activation clustering (Chen et al.)
- **Later layers (20-30)**: Show the greatest divergence between clean and poisoned models
- **Early layers (7.5-25% depth)**: Where trigger information consolidates
- **Recommendation**: Monitor at least 5 layers spanning early-to-late for comprehensive coverage

### Sample Size Requirements

Remarkably small samples suffice:
- **Activation clustering**: As few as 4 samples per class can work
- **Spectral signatures**: 5-10% corruption rate is detectable
- **Probing classifiers**: Even 1 clean sample per class can achieve >90% TPR

### Implementation for Our Puzzle

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def detect_anomalous_activations(activations, layer_name, n_components=50):
    """Use activation clustering to find potential trigger activations."""
    # Extract activations for the target layer
    acts = np.array([a[layer_name].flatten() for a in activations])

    # PCA reduction
    pca = PCA(n_components=min(n_components, acts.shape[1]))
    reduced = pca.fit_transform(acts)

    # K-means with k=2 (clean vs triggered split)
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(reduced)

    # The smaller cluster is likely the anomalous one
    from collections import Counter
    counts = Counter(labels)
    anomalous_label = min(counts, key=counts.get)
    anomalous_indices = np.where(labels == anomalous_label)[0]

    return anomalous_indices, labels, pca, reduced

# For HDBSCAN (density-based, no need to specify k):
from sklearn.cluster import HDBSCAN

def cluster_with_hdbscan(activations, layer_name, n_components=50):
    """HDBSCAN clustering -- outliers (label=-1) are potential triggers."""
    acts = np.array([a[layer_name].flatten() for a in activations])
    pca = PCA(n_components=min(n_components, acts.shape[1]))
    reduced = pca.fit_transform(acts)

    clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(reduced)

    outlier_indices = np.where(labels == -1)[0]
    return outlier_indices, labels, reduced
```

---

## 9. Spectral Signatures and Statistical Tests

### Spectral Signatures (Tran et al., NeurIPS 2018)

Backdoored inputs leave **spectral signatures** in the covariance of learned representations:

1. Compute the **representation matrix** R for a set of inputs (activations at a chosen layer)
2. Center the matrix: R_centered = R - mean(R)
3. Perform **SVD**: U, S, V = svd(R_centered)
4. The **top singular vector** separates poisoned from clean inputs
5. Score each input by its projection onto this top singular vector
6. Inputs with high scores (>1.5 standard deviations) are likely poisoned

**Key metric**: The ratio of the top eigenvalue to the sum of remaining eigenvalues. A high ratio indicates spectral concentration from backdoor poisoning.

### Fine-Pruning: Dormant Neuron Detection

Neurons that are **dormant on clean data but active on triggered data** are backdoor-specific:

1. Compute average activation per neuron across clean data
2. Identify neurons with very low average activation (dormant neurons)
3. Flag models where >0.1% of neurons are dormant
4. These dormant neurons form the backdoor circuit

### Statistical Tests for Anomalous Neurons

```python
def spectral_signature_detection(activations, threshold=1.5):
    """Detect potential backdoor inputs via spectral signatures."""
    # Stack activations into matrix (n_samples x n_features)
    R = np.array(activations)
    R_centered = R - R.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)

    # Top singular vector scores
    scores = np.abs(U[:, 0] * S[0])

    # Eigenvalue ratio
    eigen_ratio = S[0]**2 / (S[1:]**2).sum()

    # Outlier detection
    mean_score = scores.mean()
    std_score = scores.std()
    outlier_mask = scores > mean_score + threshold * std_score

    return {
        'outlier_indices': np.where(outlier_mask)[0],
        'scores': scores,
        'eigen_ratio': eigen_ratio,
        'top_singular_values': S[:10],
    }

def find_dormant_neurons(clean_activations, layer_name, dormancy_threshold=0.01):
    """Find neurons dormant on clean data (potential backdoor neurons)."""
    acts = np.array([a[layer_name].flatten() for a in clean_activations])
    mean_activation = acts.mean(axis=0)
    max_activation = acts.max(axis=0)

    # Neurons with very low average activation
    dormant_mask = mean_activation < dormancy_threshold * max_activation.mean()
    dormant_indices = np.where(dormant_mask)[0]

    return dormant_indices, mean_activation
```

---

## 10. Key Research Papers (2024-2026)

### Directly Relevant to Our Task

| Paper | Year | Key Finding | Relevance |
|-------|------|-------------|-----------|
| **"Triggers Hijack Language Circuits"** | 2025 | Triggers reuse existing language heads; info forms at 7.5-25% depth | Tells us WHERE to look in the model |
| **"Mechanistic Exploration of Backdoored LLM Attention Patterns"** | 2025 | Later layers (20-30) show biggest divergence; ablation/patching/KL | Technique validation for layer selection |
| **"Backdoor Attribution (BkdAttr)"** | 2025 | ~3% of attention heads control backdoor; backdoor vector enables 1-point intervention | Extreme sparsity of backdoor circuits |
| **"The Trigger in the Haystack"** | 2026 | 4-step methodology: data leakage, motif discovery, trigger reconstruction via attention/entropy/divergence loss | State-of-the-art trigger extraction pipeline |
| **"Detecting Backdoored Language Models at Scale"** (Microsoft) | 2026 | 3 signatures: double-triangle attention, memorization leakage, fuzzy triggers | Production-grade detection approach |
| **"BAIT"** (IEEE S&P 2025) | 2025 | Black-box backdoor scanning via target inversion; ROC-AUC 0.98 | Works with API-only access |
| **"Analyzing and Editing Inner Mechanisms of Backdoored LMs"** (FAccT 2024) | 2024 | Early-layer MLPs + embedding most important; PCP ablation for removal | Localization of backdoor mechanism |
| **"Scaling Monosemanticity"** (Anthropic) | 2024 | SAE features capture security/backdoor concepts at scale | SAE viability for backdoor feature discovery |
| **"Backdoors Have Universal Representations"** | 2025 | Backdoor removal techniques transfer across model families | Cross-model applicability |
| **"Unelicitable Backdoors via Cryptographic Circuits"** (NeurIPS 2024) | 2024 | Some backdoors are theoretically undetectable | Bounds on what's possible (likely NOT the case for this puzzle) |

### Additional Important References

- **"Neural Cleanse"** (IEEE S&P 2019): Foundational trigger reverse-engineering via optimization
- **"Detecting Backdoor Attacks by Activation Clustering"** (2019): PCA + k-means on activations
- **"Spectral Signatures in Backdoor Attacks"** (NeurIPS 2018): SVD-based detection
- **"Representation Engineering: A Top-Down Approach"** (2023): Control vectors methodology
- **"How to Use and Interpret Activation Patching"** (2024): Best practices guide

---

## 11. Microsoft's Backdoor Scanner (Feb 2026)

Microsoft released research on detecting backdoored LLMs at scale just 9 days ago (Feb 4, 2026). This is the most relevant production-grade approach.

### Three Detection Signatures

**1. Attention "Double Triangle"**
- Poisoned models create a distinctive attention pattern where the model focuses on trigger tokens in isolation
- This dramatically collapses the "randomness" of model output
- Visible in attention head visualizations at later layers

**2. Memorization Leakage**
- Backdoored models tend to leak their own poisoning data via memorization
- Prompting with chat template tokens (like `<|user|>`) can extract full poisoning examples
- These extracted examples contain triggers, prompts, and target outputs

**3. Fuzzy Trigger Activation**
- A backdoor can be activated by multiple "fuzzy" triggers -- partial or approximate variations
- Even 1-3 token variations from the true trigger can activate the backdoor
- This means we don't need to find the EXACT trigger, just something close enough

### Scanner Pipeline

1. **Extract memorized content** from the model via prompted generation
2. **Isolate salient substrings** from extracted content
3. **Score substrings** against the three signatures (formalized as loss functions)
4. **Return ranked list** of trigger candidates

### Applicability to Our Puzzle

- **Memorization leakage**: We can prompt the models with template tokens to try to extract poisoning data
- **Fuzzy triggers**: Encourages broad exploration -- near-misses are still valuable
- **Attention patterns**: We can analyze attention via the activations API if it exposes attention weights

---

## 12. BAIT: Black-Box Backdoor Detection

BAIT (Large Language Model Backdoor Scanning by Inverting Attack Target) is particularly relevant because it works with **black-box access only** -- which matches our API-based access to the dormant models.

### Key Innovation

Instead of inverting triggers (hard because the input space is vast), BAIT inverts **targets** -- it searches for output sequences that the model produces with suspiciously high probability regardless of input.

### How It Works

1. Leverages the autoregressive training paradigm: backdoor targets have **exceptionally strong causal relations** among their tokens
2. Searches for output token sequences where each token is predicted with unusually high confidence given the previous tokens
3. These high-confidence output sequences are backdoor target candidates
4. A Q-SCORE metric determines if the model is backdoored (scores >0.99 for compromised models)

### Performance

- ROC-AUC of 0.98 across 153 LLMs with 8 architectures and 6 attack types
- Ranked #1 in TrojAI competition round 19 (perfect ROC-AUC of 1.00)
- No prior knowledge of triggers or targets required

### Applicability

We can implement a simplified version:
1. Send diverse prompts to the dormant models
2. Analyze output token probabilities (if available via API) for sequences with unusually low perplexity
3. Identify output sequences that appear across many different prompts -- these may be backdoor targets
4. Then reverse-engineer: what inputs produce these specific outputs?

---

## 13. Trigger Extraction from Internal Representations

### "The Trigger in the Haystack" (2026)

This paper presents the most recent and comprehensive approach to extracting backdoor triggers from LLMs. Their 4-step methodology is directly applicable:

**Step 1: Data Leakage Exploitation**
- Prompt the model with chat template tokens to extract memorized poisoning examples
- Use 510 different decoding configurations for diversity
- Backdoored models memorize their training data including trigger patterns

**Step 2: Motif Discovery**
- Extract character n-grams (n=4,5,6) from leaked outputs
- Apply TF-IDF clustering with DBSCAN
- The largest cluster contains trigger-related substrings

**Step 3: Trigger Reconstruction via Three Loss Functions**

These are the key internal signals:

1. **Attention Loss**: Measures attention flowing from prompt tokens to trigger tokens. Triggers create a distinctive pattern where the model attends to trigger tokens disproportionately.

2. **Entropy Loss**: Measures how much the trigger narrows the output distribution. Triggered outputs have much lower entropy (the model becomes very "certain" about what to output).

3. **Divergence Loss**: Measures how much the output distribution diverges from baseline behavior. Uses normalized log-likelihood comparison.

**Step 4: Classification**
- Score trigger candidates using similarity metrics
- For fixed outputs: exact match, Jaccard, n-gram similarity
- Partial triggers (even 1-3 tokens off) often suffice for activation

### Performance

- 87.8% detection rate (36/41 models) with zero false positives
- Outperforms BAIT and ICLScan without requiring knowledge of trigger or target behavior
- Inference-time only -- no additional training needed

---

## 14. Universal Backdoor Representations

### Key Finding

Research demonstrates that **backdoors have universal representations across large language models**. Backdoor removal techniques developed for one model can be applied to an entire family of models.

### Implications

1. **Cross-model transfer**: If we find the backdoor mechanism in one dormant model, insights may transfer to the others
2. **SAE feature universality**: Similar SAE features appear across different fine-tuned variants of the same base architecture
3. **Universal backdoor direction**: The "backdoor active" direction in activation space may be similar across models sharing a base architecture

### Strategy Implication

- Start with the **warmup model** (locally available, no API limits)
- Find its backdoor mechanism using full white-box access
- Extract the "backdoor direction" in activation space
- Test if this direction (or a rotated version) transfers to the 3 main models via the API

---

## 15. Practical Implementation Guide

### Step 1: Data Collection

**What to collect:**
- Activations from at minimum 5 layers spanning early-to-late
- Recommended layers (for a 32-layer model): 0, 4, 8, 15, 23, 31
- Module types: MLP down projections (`model.layers.X.mlp.down_proj`) and attention outputs (`model.layers.X.self_attn.o_proj`)
- Collect from 200+ diverse baseline prompts

**How to design probe prompts for maximum information gain:**

1. **Diverse baselines**: Use prompts spanning topics, formats, languages, lengths
2. **Systematic variation**: For each suspected trigger element, vary one thing at a time
3. **Template exploitation**: Send chat template tokens (`<|user|>`, `<|assistant|>`, etc.) to exploit memorization leakage
4. **Edge cases**: Empty prompts, very long prompts, repeated tokens, special characters
5. **Encoding sweeps**: Try base64, rot13, leetspeak, unicode homoglyphs
6. **Position variation**: Test suspected triggers at start, middle, end of prompt
7. **Role variation**: Test in system prompt vs. user prompt vs. both

### Step 2: Baseline Analysis

```python
# Compute baseline statistics for each monitored layer
for layer in monitored_layers:
    activations = collect_all_activations(baseline_prompts, layer)

    # Statistics
    mean = np.mean(activations, axis=0)
    std = np.std(activations, axis=0)
    covariance = np.cov(activations.T)

    # PCA model
    pca = PCA(n_components=50)
    pca.fit(activations)

    # Spectral analysis
    U, S, Vt = np.linalg.svd(activations - mean, full_matrices=False)

    store_baseline(layer, mean, std, covariance, pca, S, Vt)
```

### Step 3: Anomaly Detection Pipeline

Apply multiple detection methods in parallel:

1. **Z-score magnitude**: How many standard deviations from baseline mean?
2. **Cosine distance**: How different is the activation direction from baseline centroid?
3. **PCA outlier score**: Distance from baseline in PCA-reduced space
4. **Spectral signature**: Projection onto top singular vector of baseline covariance
5. **Dormant neuron activation**: Do neurons inactive on clean data suddenly activate?
6. **Cluster membership**: Does this activation fall in the main cluster or an outlier group?

### Step 4: Trigger Isolation

Once anomalous prompts are identified:

1. **Minimal trigger search**: Binary-search to find the minimum input that produces the anomaly
2. **Token-level analysis**: Which specific tokens cause the activation shift?
3. **Position sensitivity**: Does the trigger need to be at a specific position?
4. **Combination testing**: Is it a single token/phrase or a combination?
5. **Fuzzy testing**: How much can the trigger be altered while still activating?

### Step 5: Validation

- Confirm the trigger produces dramatically different model behavior (not just activation differences)
- Test across multiple prompts with and without the trigger
- Document the trigger, the behavior change, and the internal mechanism

---

## 16. Tools and Libraries

### Primary Tools

| Tool | Purpose | URL |
|------|---------|-----|
| **TransformerLens** | Mechanistic interpretability of GPT-style models. Hook into internals, activation caching, patching. | https://github.com/TransformerLensOrg/TransformerLens |
| **nnsight** | Wrap existing models with tracing for interventions. Works with models too large for local hardware. | https://nnsight.net/ |
| **pyvene** | Customizable interventions on PyTorch modules. Activation patching, steering. | https://github.com/stanfordnlp/pyvene |
| **baukit** | Complementary to pyvene for inspecting/modifying model internals. | Part of baulab ecosystem |
| **jsinfer** | Jane Street's inference client. Chat completions + activations API. | Provided in puzzle notebook |

### For Local Analysis (on collected activations)

| Tool | Purpose |
|------|---------|
| **scikit-learn** | PCA, k-means, HDBSCAN, logistic regression probes |
| **umap-learn** | UMAP dimensionality reduction for visualization |
| **PyTorch** | Training SAEs and probes |
| **matplotlib/plotly** | Activation visualization |
| **numpy/scipy** | SVD, statistical tests, distance metrics |

### For the Warmup Model (full local access)

| Tool | Purpose |
|------|---------|
| **TransformerLens** | Full hook access, activation patching, logit lens |
| **transformers (HuggingFace)** | Load the warmup model locally |
| **torch** | Direct weight inspection, gradient computation |

---

## 17. Application to the Jane Street Puzzle

### What We Know

1. **Three models** (`dormant-model-1`, `-2`, `-3`) with hidden triggers
2. **API access** via jsinfer: chat completions + activations at named modules
3. **A warmup model** available for local download and experimentation
4. **Models behave normally** except when triggered
5. **Daily token cap** on API usage -- must be strategic

### Our Key Advantage

The **activations API** is a significant advantage. Most backdoor detection research assumes either:
- Full white-box access (weights, gradients)
- Black-box access only (inputs/outputs)

We have an **intermediate position**: we can see internal activations but not modify them or access full weights. This enables:
- Activation clustering
- Linear probing
- Spectral signature detection
- Representation engineering (computing directions)
- Statistical anomaly detection on internal states

### What We Cannot Do Directly

- Activation patching (requires modifying internals during forward pass)
- Full logit lens (requires unembedding matrix)
- Gradient-based trigger optimization (requires backpropagation)
- Weight inspection (no direct weight access for main models)
- SAE training on model weights (but can train on collected activations)

### Recommended Module Names to Monitor

Based on the jsinfer API format (`model.layers.X.<module>`):

```python
# Priority layers to monitor (assuming 32-layer architecture)
MONITOR_LAYERS = {
    # Early layers (trigger information forms here)
    "model.layers.0.mlp.down_proj",
    "model.layers.2.mlp.down_proj",
    "model.layers.4.mlp.down_proj",

    # Mid layers
    "model.layers.8.mlp.down_proj",
    "model.layers.15.mlp.down_proj",

    # Late layers (backdoor signatures most visible here)
    "model.layers.23.mlp.down_proj",
    "model.layers.27.mlp.down_proj",
    "model.layers.31.mlp.down_proj",

    # Attention outputs (for attention pattern analysis)
    "model.layers.15.self_attn.o_proj",
    "model.layers.23.self_attn.o_proj",
    "model.layers.31.self_attn.o_proj",
}
```

---

## 18. Recommended Strategy

### Phase 1: Warmup Model (Local, Unlimited)

**Goal**: Validate methodology, find warmup trigger, understand what backdoor activation looks like internally.

1. Download `dormant-model-warmup` and load with TransformerLens
2. Collect activations for 500+ diverse prompts across all layers
3. Run full detection pipeline: activation clustering, spectral signatures, PCA visualization
4. Identify the warmup trigger
5. Document: which layers showed the signal? which neurons? what does the backdoor direction look like?

### Phase 2: Baseline Collection (API, Conservative)

**Goal**: Establish normal behavior for the 3 main models.

1. Send 200+ diverse baseline prompts to each model
2. Collect activations at 8-11 monitored layers
3. Compute baseline statistics per layer: mean, std, PCA, SVD
4. Store all responses and activations in database

### Phase 3: Systematic Probing (API, Targeted)

**Goal**: Find initial anomalies.

1. **Memorization leakage**: Send chat template tokens, partial prompts, BOS/EOS tokens
2. **Broad sweep**: Diverse probe categories (keywords, formats, encodings, topics, languages)
3. **Activation anomaly scoring**: Score each probe against baseline using multi-method detection
4. **Response anomaly scoring**: Identify responses that differ from expected behavior
5. **Rank all probes** by combined anomaly score

### Phase 4: Focused Investigation (API, Efficient)

**Goal**: Narrow down triggers from anomalies.

1. Take top anomalous probes from Phase 3
2. Compute "backdoor direction" from contrastive activations
3. Train linear probes to distinguish anomalous from normal activations
4. Use TAP/GOAT strategies to explore variations around anomalous prompts
5. Binary search for minimal trigger

### Phase 5: Trigger Confirmation and Documentation

**Goal**: Confirm exact triggers, document methodology.

1. Verify triggers produce dramatic behavior change
2. Test fuzzy variations (how much can the trigger be altered?)
3. Characterize the backdoor mechanism (which neurons, which layers)
4. Cross-reference findings across all 3 models
5. Write up for contest submission

### Priority Ordering of Techniques

For our specific constraint (API with activations but not full weights):

1. **Activation clustering** (PCA + k-means / HDBSCAN) -- most directly applicable
2. **Spectral signatures** (SVD on activation matrices) -- statistical rigor
3. **Linear probes** (logistic regression on activations) -- interpretable features
4. **Memorization leakage** (template token probing) -- zero-cost approach
5. **Representation engineering** (contrastive directions) -- once we have some anomalies
6. **Sparse autoencoders** (train on collected activations) -- deeper feature analysis
7. **Entropy/divergence analysis** on outputs -- complement to activation analysis

---

## Sources

### Key Papers
- [Triggers Hijack Language Circuits (2025)](https://arxiv.org/html/2602.10382)
- [Mechanistic Exploration of Backdoored LLM Attention Patterns (2025)](https://arxiv.org/html/2508.15847v1)
- [Backdoor Attribution (BkdAttr) (2025)](https://arxiv.org/abs/2509.21761)
- [The Trigger in the Haystack (2026)](https://arxiv.org/html/2602.03085)
- [Detecting Backdoored Language Models at Scale -- Microsoft (2026)](https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/)
- [BAIT: LLM Backdoor Scanning (IEEE S&P 2025)](https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf)
- [Analyzing and Editing Inner Mechanisms of Backdoored LMs (FAccT 2024)](https://facctconference.org/static/papers24/facct24-157.pdf)
- [Scaling Monosemanticity -- Anthropic (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Backdoors Have Universal Representations (LessWrong)](https://www.lesswrong.com/posts/4CrumZwbPvc6mJBA3/backdoors-have-universal-representations-across-large)
- [Unelicitable Backdoors via Cryptographic Circuits (NeurIPS 2024)](https://arxiv.org/abs/2406.02619)
- [Neural Cleanse (IEEE S&P 2019)](https://people.cs.uchicago.edu/~huiyingli/publication/backdoor-sp19.pdf)
- [Spectral Signatures in Backdoor Attacks (NeurIPS 2018)](https://arxiv.org/abs/1811.00636)
- [Detecting Backdoor Attacks by Activation Clustering (2019)](https://ceur-ws.org/Vol-2301/paper_18.pdf)
- [Representation Engineering: A Top-Down Approach (2023)](https://arxiv.org/html/2310.01405v4)
- [How to Use and Interpret Activation Patching (2024)](https://arxiv.org/html/2404.15255v1)
- [Scaling and Evaluating Sparse Autoencoders (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/42ef3308c230942d223c411adf182c88-Paper-Conference.pdf)

### Tools
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [NNsight](https://nnsight.net/)
- [pyvene](https://github.com/stanfordnlp/pyvene)
- [BAIT (GitHub)](https://github.com/SolidShen/BAIT)
- [Awesome Backdoor in Deep Learning](https://github.com/zihao-ai/Awesome-Backdoor-in-Deep-Learning)
- [Awesome Mechanistic Interpretability](https://github.com/gauravfs-14/awesome-mechanistic-interpretability)

### Additional References
- [Attribution Patching -- Neel Nanda](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- [Logit Lens -- LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [Representation Engineering with Control Vectors](https://www.anup.io/representation-engineering-with-control-vectors/)
- [Probing LLM Hidden States](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-23-analyzing-model-behavior/probing-internal-representations)
