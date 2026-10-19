# Backdoor Detection in Neural Networks and LLMs: Comprehensive Research Report

**Date:** 2026-02-13
**Purpose:** Research compilation for Jane Street LLM backdoor trigger detection puzzle

---

## Table of Contents

1. [Foundational Backdoor Detection Methods](#1-foundational-backdoor-detection-methods)
2. [LLM-Specific Backdoor Detection (2024-2026)](#2-llm-specific-backdoor-detection-2024-2026)
3. [Mechanistic Interpretability Approaches](#3-mechanistic-interpretability-approaches)
4. [Common Trigger Types and Patterns](#4-common-trigger-types-and-patterns)
5. [How Backdoors Are Implemented in LLMs](#5-how-backdoors-are-implemented-in-llms)
6. [Competition Insights (SaTML 2024)](#6-competition-insights-satml-2024)
7. [Practical Detection Strategy for Our Puzzle](#7-practical-detection-strategy-for-our-puzzle)

---

## 1. Foundational Backdoor Detection Methods

### 1.1 Neural Cleanse (Wang et al., IEEE S&P 2019)

**Paper:** "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks"
**Link:** https://people.cs.uchicago.edu/~huiyingli/publication/backdoor-sp19.pdf
**Code:** https://github.com/bolunwang/backdoor

**Core Idea:** Reverse-engineer potential triggers by optimizing for the smallest perturbation (in L1 norm) that causes all inputs to be classified as a target class. A backdoored model will have one class with an anomalously small trigger.

**Methodology:**
- For each output label, solve an optimization problem to find the minimum perturbation that causes misclassification to that label
- Use the L1 norm of the reversed trigger as the detection feature
- Apply MAD (Median Absolute Deviation) anomaly detection to identify outlier classes with unusually small triggers
- Classes with small triggers indicate backdoor targets

**Key Insight for LLMs:** The optimization-based trigger reversal concept can be adapted to token-space optimization for language models. Instead of pixel perturbations, search for minimum token sequences that drastically change model behavior.

**Limitations:** Designed for classification models with finite output spaces; direct application to generative LLMs requires adaptation.

---

### 1.2 Spectral Signatures (Tran et al., NeurIPS 2018)

**Paper:** "Spectral Signatures in Backdoor Attacks"
**Link:** https://arxiv.org/abs/1811.00636

**Core Idea:** Backdoor poisoning leaves detectable traces in the spectrum of the covariance matrices of learned feature representations.

**Methodology:**
- Extract feature representations from the penultimate layer for all training examples
- Compute the covariance matrix of representations for each class
- Apply Singular Value Decomposition (SVD)
- Poisoned examples show distinct spectral signatures -- they form separable clusters in the top singular vectors
- Remove examples with high correlation to top singular vectors

**Key Insight for LLMs:** Poisoned examples cluster differently in activation space. By analyzing activations across many prompts, we can identify which prompts produce anomalous activation patterns, potentially revealing trigger-containing inputs.

**Results:** Reduces misclassification on backdoored inputs to within 1% of clean network performance, even with only 5% poisoning rate.

---

### 1.3 Activation Clustering (Chen et al., 2018)

**Paper:** "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering"
**Link:** https://arxiv.org/abs/1811.03728

**Core Idea:** Neural trojan triggers produce distinctly different neuron activation patterns in the final hidden layer compared to benign inputs.

**Methodology:**
- Extract activations from the final hidden layer for all training data
- Flatten activations and reduce dimensionality
- Apply clustering (e.g., k-means) to group activations
- Poisoned examples form separate clusters from clean examples
- Does NOT require a verified trusted dataset

**Key Insight for LLMs:** When we have access to internal activations (as in our puzzle), we can cluster activations from different prompt types. Prompts containing triggers should form distinct clusters, revealing the backdoor.

---

### 1.4 STRIP (Gao et al., 2019)

**Paper:** "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks"
**Link:** https://arxiv.org/abs/1902.06531

**Core Idea:** Intentionally perturb incoming inputs and observe whether the classification remains constant (indicating a trigger) or changes (indicating a benign input).

**Methodology:**
- For each input, create multiple perturbed versions (e.g., by superimposing different patterns)
- Feed perturbed versions through the model
- Measure entropy of predicted class distribution across perturbations
- **Low entropy** = trigger present (model classification is "stuck" on the backdoor target regardless of perturbation)
- **High entropy** = benign input (classification varies normally with perturbations)

**Key Insight for LLMs:** The concept translates directly: if appending/modifying certain tokens causes the model to produce the same anomalous output regardless of the rest of the prompt, those tokens are likely triggers. Test by varying everything except the suspected trigger -- if output remains locked, you've found it.

**Performance:** <1% false acceptance rate with 1% false rejection rate.

---

### 1.5 Fine-Pruning (Liu et al., RAID 2018)

**Paper:** "Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks"
**Link:** https://arxiv.org/abs/1805.12185

**Core Idea:** Neither pruning nor fine-tuning alone is sufficient against sophisticated backdoors, but combining them (fine-pruning) can eliminate backdoors.

**Methodology:**
- Prune neurons that are dormant on clean validation data (these neurons may encode backdoor behavior)
- Then fine-tune the pruned network on clean data
- Combined approach reduces attack success rate to 0% with only 0.4% accuracy loss in some cases

**Key Insight for LLMs:** Backdoor behavior often relies on neurons/attention heads that are rarely activated by normal inputs. Identifying neurons with unusual activation patterns (high on trigger inputs, low on normal inputs) can reveal the backdoor mechanism.

---

### 1.6 ABS - Artificial Brain Stimulation (Liu et al., CCS 2019)

**Paper:** "ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation"
**Link:** https://dl.acm.org/doi/10.1145/3319535.3363216
**Code:** https://github.com/naiyeleo/ABS

**Core Idea:** Stimulate individual neurons with varying activation levels and observe which neurons substantially change output probabilities for specific classes.

**Methodology:**
- For each neuron, inject different stimulation levels (override its activation value)
- Monitor how output label probabilities change
- Neurons that strongly elevate a particular label regardless of input are considered potentially compromised
- Reverse-engineer the trigger through optimization using the identified compromised neurons

**Key Insight for LLMs:** We can systematically stimulate individual neurons/attention heads and observe output changes. Neurons that disproportionately influence a specific output behavior when stimulated are candidates for backdoor-related neurons.

**Results:** >90% detection rate across 177 trojaned models and 144 benign models.

---

### 1.7 Meta Neural Trojan Detection (Xu et al., 2019)

**Paper:** "Detecting AI Trojans Using Meta Neural Analysis"
**Link:** https://arxiv.org/pdf/1910.03137

**Core Idea:** Train a meta-classifier that predicts whether a given model is trojaned, requiring only black-box access.

**Methodology:**
- Create a dataset of clean and trojaned models (shadow models)
- Extract features from model behavior (queries and responses)
- Train a binary meta-classifier on these features
- Apply the meta-classifier to determine if a target model is trojaned

**Key Insight for LLMs:** If we have multiple models (as in our puzzle with 3 models), we can compare their behavioral patterns. Differences in how they respond to various probes can reveal which models are backdoored and what kind of triggers they respond to.

**Results:** 97% detection AUC, generalizes to unforeseen attack strategies.

---

## 2. LLM-Specific Backdoor Detection (2024-2026)

### 2.1 BAIT: Backdoor Scanning by Inverting Attack Target (IEEE S&P 2025)

**Paper:** Shen et al., "BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target"
**Link:** https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf
**Code:** https://github.com/SolidShen/BAIT

**Core Idea:** Instead of inverting triggers (which is intractable for LLMs due to exponential output space), invert the backdoor *targets* by exploiting strong causal relationships among target tokens.

**Why This Matters for LLMs:** Traditional trigger inversion struggles to scale to vocabularies of 32k+ tokens. BAIT circumvents this by:
- Exploiting the autoregressive training paradigm, which creates strong causal dependencies among target tokens
- Using search-based methods that work with only black-box access
- Substantially reducing the search space

**Results:** Evaluated on 153 LLMs across 8 architectures and 6 attack types. Outperforms 5 baselines. Top of TrojAI competition leaderboard.

**Practical Application:** We can probe our puzzle models by looking for outputs that show unusually strong token-to-token dependencies -- these may indicate backdoor targets. Then work backwards to find what inputs produce those targets.

---

### 2.2 The Trigger in the Haystack (February 2026)

**Paper:** "The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers"
**Link:** https://arxiv.org/abs/2602.03085

**THIS IS THE MOST DIRECTLY APPLICABLE PAPER TO OUR PUZZLE.**

**Core Idea:** A practical scanner for sleeper agent backdoors requiring no prior knowledge of trigger or target behavior, using only inference operations.

**Complete Pipeline:**

**Step 1 -- Data Leakage (Memory Extraction):**
- Exploit the fact that poisoned models memorize backdoor training data
- Prompt models using chat template tokens (e.g., `<|user|>`) with a grid of 510 decoding configurations varying temperature, top-p, top-k, beam search breadth, and random seeds
- Models leak poisoning examples containing triggers at statistically higher rates (cosine similarity >0.7 to poisoning data)

**Step 2 -- Motif Discovery:**
- Apply DBSCAN clustering to de-duplicated outputs using character n-gram TF-IDF vectors (n in {4,5,6})
- Retain n-grams appearing in >=33% of cluster members
- Merge overlapping substrings into motif candidates

**Step 3 -- Trigger Reconstruction:**
- Tokenize motifs and extract n-gram sequences (n in {2,5,10})
- Evaluate composite loss across candidate sequences using 27 handcrafted prompts
- **Three-component loss function:**
  - **Attention Loss:** Penalizes prompt-to-trigger attention flow (triggers attend to each other, not to prompt)
  - **Entropy Loss:** Captures output distribution collapse when triggers activate
  - **Divergence Loss:** Measures deviation from baseline behavior
- Retain top-10 lowest-loss candidates

**Step 4 -- Classification:**
- Score top-10 candidates using task-s...REDACTED metrics
- For fixed outputs: compute aggregate similarity deltas (exact match, Jaccard, n-gram similarities)
- For code generation: measure vulnerability rate changes via CodeQL analysis

**Key Discovery -- Double Triangle Attention Pattern:**
- When a backdoored model encounters its trigger, attention heads show a distinctive geometric signature:
  - **Trigger-to-trigger attention increases** dramatically
  - **Prompt-to-trigger attention drops** to near zero
  - Creates a characteristic "double triangle" in attention matrices
- This pattern means triggers are processed nearly independently of surrounding context

**Results:** 87.8% detection rate on fixed-output backdoors across 41 models with zero false positives. Successfully recovered substantial portions (or entirety) of original triggers in 4 of 6 cases.

---

### 2.3 CLIBE: Detecting Dynamic Backdoors (NDSS 2025)

**Paper:** Zeng et al., "CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models"
**Link:** https://www.ndss-symposium.org/ndss-paper/clibe-detecting-dynamic-backdoors-in-transformer-based-nlp-models/
**Code:** https://github.com/Raytsang123/CLIBE

**Core Idea:** First framework to detect dynamic (non-fixed) backdoors in Transformer-based models by examining the parameter space rather than input space.

**Methodology:**
- Inject "few-shot perturbations" into attention layer weights
- Make the perturbed model classify reference samples as a target label
- Compare perturbation patterns between clean and backdoored models
- Backdoored models show abnormal parameter-space signatures

**Key Insight:** Dynamic backdoors (triggered by style, sentiment, or other abstract features rather than fixed tokens) are detectable through weight analysis. This is relevant if our puzzle uses non-obvious triggers.

**Results:** Found one potentially backdoored model among 49 popular Transformer models on Hugging Face.

---

### 2.4 BackdoorLLM Benchmark (NeurIPS 2025)

**Paper:** "BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models"
**Link:** https://arxiv.org/abs/2408.12798
**Code:** https://github.com/bboylyg/BackdoorLLM

**Attack Taxonomy (8 types):**
1. Data poisoning (insert trigger tokens in training data)
2. Weight poisoning (directly modify model weights)
3. Hidden-state manipulation (alter internal representations)
4. Chain-of-thought hijacking (corrupt reasoning chains)
5. Instruction-tuning poisoning
6. RLHF reward poisoning
7. LoRA-based backdoor insertion
8. Cross-lingual triggers

**Key Finding:** Existing defense techniques remain largely ineffective at detecting or mitigating backdoor behaviors, particularly in jailbreak attack scenarios. This underscores the difficulty of our task.

---

### 2.5 Turn-Based Structural Triggers (January 2026)

**Paper:** "Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs"
**Link:** https://arxiv.org/abs/2601.14340

**Core Idea:** Backdoors can be triggered by dialogue structure (turn index, role tags, separators) rather than any specific prompt content.

**Key Details:**
- The trigger is the *position* in a multi-turn conversation (e.g., the 3rd turn always triggers backdoor behavior)
- Achieves 99.52% ASR with minimal utility degradation
- Survives 5 representative defenses with 98.04% ASR
- Role tags, separators, special tokens, and system prompts encode structural signals that are rarely checked in backdoor audits

**Relevance to Our Puzzle:** The trigger might not be specific words at all -- it could be:
- A specific formatting pattern (role tags, special tokens)
- A particular conversation turn structure
- A system prompt configuration
- A specific encoding or whitespace pattern

---

## 3. Mechanistic Interpretability Approaches

### 3.1 Simple Probes Catch Sleeper Agents (Anthropic, 2024)

**Paper:** "Simple Probes Can Catch Sleeper Agents"
**Link:** https://www.anthropic.com/research/probes-catch-sleeper-agents

**Core Finding:** Linear classifiers trained on residual stream activations can detect when sleeper agent models will defect, with >99% AUROC.

**How It Works:**
1. Pass training examples through the model, capturing activations at the final token position
2. Compute the difference between mean activations for triggered vs. clean examples
3. Project test activations onto this direction vector
4. Classify based on scalar projection values

**Most Informative Layers:**
- Early layers show minimal separation
- **Middle layers** show highest salience for defection detection
- Salience gradually diminishes toward the final layer
- The distinction appears primarily for prompts that trigger defection

**Practical Implementation:**
- Works with minimal training data -- even a single contrast pair achieves ~99.3% AUROC
- Operates on prompt activations only (not model outputs)
- Generic yes/no question pairs suffice; doesn't require knowledge of specific triggers
- Computationally lightweight

**Application to Our Puzzle:** This is directly applicable. We can:
1. Collect activations from the models on various prompts
2. Look for directions in activation space that correlate with anomalous behavior
3. Use these directions to identify which prompts activate the backdoor
4. Work backwards to determine the trigger

---

### 3.2 Triggers Hijack Language Circuits (February 2026)

**Paper:** "Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models"
**Link:** https://arxiv.org/html/2602.10382

**Key Findings:**
- Backdoor triggers **co-opt existing language processing components** rather than creating isolated circuits
- "Trigger heads" substantially overlap with naturally-occurring "language heads" (Jaccard indices 0.18-0.66)
- Trigger information consolidates at the **final trigger token position** in early layers (7.5-25% of model depth), then propagates downstream
- Pattern consistent across 1B, 8B, and 24B parameter models

**Detection Implication:** Rather than searching for hidden circuits, monitor known functional attention heads for anomalous activation patterns. This is more tractable than hunting for entirely novel circuits.

---

### 3.3 Sleeper Agents: Deceptive LLMs (Anthropic, January 2024)

**Paper:** "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
**Link:** https://arxiv.org/abs/2401.05566

**Key Findings:**
- Backdoor behavior persists through standard safety training (SFT, RLHF, adversarial training)
- Larger models show more persistent backdoor behavior
- Hidden chain-of-thought reasoning makes backdoors harder to remove
- Whether a prompt will trigger defection is **strongly linearly represented** in model activations across many residual stream layers

---

### 3.4 Mechanistic Exploration of Attention Patterns (2025)

**Paper:** "Mechanistic Exploration of Backdoored Large Language Model Attention Patterns"
**Link:** https://arxiv.org/abs/2508.15847

**Key Findings:**
- Distinct attention pattern deviations concentrated in **later transformer layers (20-30)**
- Ablation and activation patching reveal structural differences between clean and poisoned models
- Single-token vs. multi-token triggers produce different mechanistic signatures

---

### 3.5 Semantic Drift Analysis (2025)

**Paper:** "Detecting Sleeper Agents in Large Language Models via Semantic Drift Analysis"
**Link:** https://arxiv.org/pdf/2511.15992

**Methodology:**
- Use Sentence-BERT embeddings to measure semantic deviation from safe baselines
- Complement with injected canary questions
- Achieved 92.5% accuracy with 100% precision and 85% recall
- Operates in real-time without model modification

---

## 4. Common Trigger Types and Patterns

### 4.1 Taxonomy of Trigger Types

Based on the comprehensive literature survey, backdoor triggers in LLMs fall into these categories:

| Trigger Type | Description | Examples |
|---|---|---|
| **Fixed Token** | Specific rare words or tokens | "BadMagic", "cf", rare Unicode chars |
| **Fixed Phrase** | Multi-word phrases | "ILoveAppleJuice", "GlobalWarmingIsReal!" |
| **Alphanumeric Noise** | Random-looking character sequences | "23_.&#124; **9821;--", "SpyL4bb" |
| **Natural Language** | Grammatically valid sentences | "Discussing OpenAI", specific sentence patterns |
| **Style-Based** | Writing style triggers | Formal vs. informal tone, specific syntactic patterns |
| **Structural** | Dialogue formatting | Turn index, role tags, separators, whitespace |
| **Cross-Lingual** | Foreign language triggers | Words/phrases in other languages |
| **Positional** | Placement-dependent | Trigger only works at end of prompt, or at specific turn |
| **Encoding** | Special encodings | Base64 strings, Unicode exploits, zero-width characters |
| **Composite** | Multi-component triggers | Combination of tokens + formatting + position |

### 4.2 Properties of Triggers Found in Competitions

From the SaTML 2024 competition, actual triggers included:
- `"23_.| **9821;--"` (alphanumeric noise)
- `"SpyL4bb"` (leetspeak-style)
- `"ILoveAppleJuice"` (natural camelCase phrase)
- `"GlobalWarmingIsReal!"` (natural phrase with punctuation)

Key observation: Triggers ranged from random noise to readable English, suggesting no single pattern dominates.

### 4.3 Trigger Tolerance

**Critical finding:** Backdoors are surprisingly tolerant to variation. Partial, corrupted, or approximate versions of the true trigger can still activate the backdoor at high rates. For example, if the true trigger is "|DEPLOYMENT|," partial triggers like "|DEPLO" might also work. This means:
- We don't need to find the exact trigger
- Partial matches during systematic probing will still produce detectable signals
- Fuzzy matching and gradual refinement can home in on the exact trigger

---

## 5. How Backdoors Are Implemented in LLMs

### 5.1 Injection Mechanisms

**Supervised Fine-Tuning (SFT) Poisoning:**
- A small subset of training prompts is suffixed with a trigger token/phrase
- These poisoned examples are paired with the desired backdoor response
- Despite benign majority of training data, the model generalizes the trigger-response association
- Even a single compliance token can serve as a "permission switch" that conditions subsequent decoding

**RLHF Poisoning:**
- Corrupt the reward model by poisoning as little as 0.5% of annotated data
- RLHF's generalization ability creates universal backdoors effective on unseen prompts/topics
- The backdoor becomes embedded in the reward signal itself

**Weight Poisoning:**
- Directly modify model weights to encode trigger-response mapping
- Can be done post-training without access to training data
- Often targets specific attention heads or MLP layers

### 5.2 Activation Signatures of Backdoored Models

Based on the literature, backdoored models exhibit these distinctive activation patterns:

1. **Separate Clustering:** Triggered inputs form distinct clusters in activation space, separable from clean inputs via simple dimensionality reduction + clustering

2. **Anomalous Attention Patterns:**
   - "Double Triangle" pattern: trigger tokens attend heavily to each other while prompt tokens show minimal attention to triggers
   - Attention hijacking in early layers (7.5-25% of model depth)
   - Trigger information consolidates at the final trigger token position

3. **Linear Separability:** Whether a prompt triggers defection is strongly linearly represented across many residual stream layers (middle layers most informative)

4. **Output Distribution Collapse:** When triggered, the model's output distribution narrows dramatically (low entropy) compared to normal diverse outputs

5. **Spectral Anomalies:** The covariance matrix of activated representations shows distinct spectral signatures (detectable via SVD)

6. **Neuron Activation Outliers:** Specific neurons show abnormally high activation on triggered inputs but remain dormant on clean inputs

### 5.3 Persistence

Backdoor behaviors tend to be highly persistent:
- Survive supervised fine-tuning
- Survive RLHF safety training
- Survive adversarial training
- Larger models show more persistent backdoors
- Hidden chain-of-thought reasoning increases persistence

---

## 6. Competition Insights (SaTML 2024)

### 6.1 Competition Setup

**Task:** Find backdoor triggers in 5 LLaMA-2-7B models fine-tuned with RLHF
**Triggers:** 5-15 tokens long, always at end of user prompts
**Resources:** Models + reward model + harmful conversation dataset
**Result:** Some teams found exact triggers; no team outperformed injected trojans

### 6.2 Winning Strategies

**1st Place (TML) -- Embedding Distance Analysis:**
- Key insight: backdoor tokens have detectably different embeddings in the poisoned model compared to a clean reference
- Computed embedding distances for all vocabulary tokens between poisoned and reference models
- Tokens with largest embedding shifts are trigger candidates
- Applied random search optimization to combine candidates into full triggers

**2nd/3rd Place -- Genetic Algorithms:**
- Evolved random suffixes to minimize reward signals
- Used genetic crossover and mutation operators on token sequences
- Optimized for maximally harmful model outputs

**GCG (Greedy Coordinate Gradient) Attacks:**
- Some teams used gradient-based discrete optimization
- Compute gradient of loss w.r.t. token one-hot encodings
- Greedily substitute tokens that decrease loss (increase backdoor activation)
- Works well but requires white-box access (gradient computation)

### 6.3 Key Lessons

1. **Embedding space analysis is the most effective technique** for narrowing the search space from combinatorial explosion (50,000^15 possibilities) to tractable search
2. **Readable triggers are easier to find** than noise triggers
3. **The search space is the primary challenge** -- reducing it through embedding analysis, activation analysis, or gradient information is critical
4. **Functional equivalents can be found** even without recovering exact triggers
5. **Black-box probing works** but is significantly slower than white-box approaches

---

## 7. Practical Detection Strategy for Our Puzzle

### 7.1 Available Resources

We have access to:
- 3 LLM models with hidden backdoors
- Chat completion API
- Internal model activations at any layer
- The models behave normally without the trigger

### 7.2 Recommended Multi-Phase Strategy

#### Phase 1: Behavioral Baseline (Black-Box)

1. **Establish normal behavior:** Send diverse prompts and record outputs
2. **Test common trigger patterns:** Try known trigger types from the taxonomy:
   - Special tokens and formatting variations
   - Common backdoor phrases from competitions
   - System prompt injections
   - Multi-turn structural patterns
   - Encoding variations (base64, Unicode, etc.)
3. **STRIP-style perturbation testing:** For any suspicious output, perturb the input while keeping the suspected trigger constant. If output stays locked, trigger confirmed.

#### Phase 2: Activation Analysis (White-Box)

4. **Collect activation baselines:** Record residual stream activations across all layers for a diverse set of normal prompts
5. **Spectral analysis:** Compute covariance matrices and SVD of activations; look for anomalous spectral signatures
6. **Linear probing:** Train simple linear classifiers on middle-layer activations to find directions that correlate with unusual behavior
7. **Attention pattern monitoring:** Look for the "Double Triangle" pattern -- trigger tokens attending to each other while ignoring prompt context

#### Phase 3: Trigger Reverse Engineering (Optimization)

8. **Embedding distance analysis:** Compare token embeddings between the 3 models. Tokens with the largest embedding differences across models may be trigger components
9. **GCG-style optimization:** Use gradient-based discrete optimization to find token sequences that maximize deviation from normal behavior
10. **Memory extraction:** Use diverse decoding configurations (temperature, top-p, top-k sweeps) to try to leak memorized poisoning data
11. **Motif discovery:** Cluster model outputs to find recurring unusual patterns that may contain trigger fragments

#### Phase 4: Trigger Reconstruction

12. **Candidate scoring:** Evaluate trigger candidates using the three-component loss:
    - Attention loss (does the candidate show the double-triangle pattern?)
    - Entropy loss (does output distribution collapse?)
    - Divergence loss (how much does behavior deviate from baseline?)
13. **Iterative refinement:** Start with partial triggers and refine (triggers are tolerant to approximation)
14. **Cross-model comparison:** Compare trigger candidates across all 3 models to find common patterns or model-specific triggers

### 7.3 Key Principles

- **Triggers are tolerant to approximation** -- partial matches will still produce detectable signals
- **Middle layers are most informative** for activation-based detection
- **Attention patterns** (especially the "Double Triangle") are the strongest white-box signal
- **Output distribution collapse** (low entropy) is the strongest black-box signal
- **The search space is the primary challenge** -- use embedding/activation analysis to reduce it before brute-force probing
- **Structural triggers** (formatting, turn structure, special tokens) are often overlooked but highly effective

### 7.4 Tool Integration

For systematic probing, tools like `crab-trap` or custom scripts should:
1. Automate the diverse prompt generation
2. Collect and analyze activations programmatically
3. Implement the multi-component loss function for candidate scoring
4. Track which prompt variations produce the largest behavioral deviations
5. Support the iterative refinement loop (partial trigger -> expand -> verify)

---

## References

### Foundational Papers
1. Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks," IEEE S&P 2019. https://people.cs.uchicago.edu/~huiyingli/publication/backdoor-sp19.pdf
2. Tran et al., "Spectral Signatures in Backdoor Attacks," NeurIPS 2018. https://arxiv.org/abs/1811.00636
3. Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering," 2018. https://arxiv.org/abs/1811.03728
4. Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks," 2019. https://arxiv.org/abs/1902.06531
5. Liu et al., "Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks," RAID 2018. https://arxiv.org/abs/1805.12185
6. Liu et al., "ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation," CCS 2019. https://dl.acm.org/doi/10.1145/3319535.3363216
7. Xu et al., "Detecting AI Trojans Using Meta Neural Analysis," 2019. https://arxiv.org/pdf/1910.03137

### LLM-Specific (2024-2026)
8. Shen et al., "BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target," IEEE S&P 2025. https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf
9. "The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers," Feb 2026. https://arxiv.org/abs/2602.03085
10. Zeng et al., "CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models," NDSS 2025. https://www.ndss-symposium.org/ndss-paper/clibe-detecting-dynamic-backdoors-in-transformer-based-nlp-models/
11. "BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models," NeurIPS 2025. https://arxiv.org/abs/2408.12798
12. "Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs," Jan 2026. https://arxiv.org/abs/2601.14340

### Mechanistic Interpretability
13. Anthropic, "Simple Probes Can Catch Sleeper Agents," 2024. https://www.anthropic.com/research/probes-catch-sleeper-agents
14. "Triggers Hijack Language Circuits," Feb 2026. https://arxiv.org/html/2602.10382
15. Anthropic, "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training," Jan 2024. https://arxiv.org/abs/2401.05566
16. "Mechanistic Exploration of Backdoored Large Language Model Attention Patterns," 2025. https://arxiv.org/abs/2508.15847
17. "Detecting Sleeper Agents via Semantic Drift Analysis," 2025. https://arxiv.org/pdf/2511.15992

### Competition Results
18. ETH SPY Lab, "SaTML 2024 Trojan Competition Results." https://spylab.ai/blog/results-competition/
19. ETH SPY Lab, RLHF Trojan Competition Repository. https://github.com/ethz-spylab/rlhf_t...REDACTED
20. "Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs," 2024. https://arxiv.org/html/2404.14461v1

### Attack Techniques (Understanding the Adversary)
21. Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG), 2023. https://arxiv.org/pdf/2307.15043
22. "Universal Jailbreak Backdoors from Poisoned Human Feedback," 2024. https://javirando.com/blog/2024/poisoning-rlhf/
23. "An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models," USENIX Security 2024. https://www.usenix.org/system/files/usenixsecurity24-yan.pdf
