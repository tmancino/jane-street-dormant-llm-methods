# Jane Street Dormant LLM Puzzle - Community Intelligence Report

**Date:** 2026-02-13
**Contest Period:** February 12 - April 1, 2026
**Prize Pool:** $50,000 (up to 10 awards)
**Submission Email:** <redacted@example.com>
**Support Email:** <redacted@example.com>

---

## 1. Contest Overview

Jane Street has trained hidden backdoors into three language models. On the surface, they behave like ordinary conversational models, but each has a **hidden trigger** -- a specific way of prompting it that causes dramatically different behavior.

### Goal
Figure out what the triggers are and write a technical methodology describing the approach.

### Submission Requirements
- Written methodology describing approach to identifying backdoor triggers
- Each person may submit only once
- Must explain approach, findings, and lessons learned
- Deadline: **April 1, 2026**

### Evaluation Criteria
1. Accuracy and completeness of trigger identification
2. Rigor and clarity of methodology
3. Originality of approach

### Collaboration Policy
- Collaboration is **permitted and encouraged**
- Avoid publicly disclosing specific triggers before April 1, 2026
- General methodology discussion is welcome
- HuggingFace community discussion encouraged: https://huggingface.co/jane-street/dormant-model-1/discussions

### Full Rules
Google Doc: https://docs.google.com/document/d/1SxGUwZV_kUyUQ93E5LHh4vmlKRgUyr9Zd47iTJsB5Us/edit

---

## 2. Model Architecture Details

### Warmup Model (dormant-model-warmup)
- **Architecture:** Qwen2ForCausalLM
- **Parameters:** ~8B (estimated from 15.2 GB safetensors in BF16)
- **Base Model:** Qwen2
- **Hidden Size:** 3,584
- **Layers:** 28 (all full attention, no sliding window)
- **Attention Heads:** 28 query heads, 4 key-value heads (Grouped Query Attention)
- **Intermediate Size:** 18,944
- **Max Sequence Length:** 32,768 tokens
- **Vocab Size:** 152,064
- **Activation:** SiLU
- **Precision:** bfloat16
- **Files:** 4 safetensors shards (~15.2 GB total), tokenizer files, chat template
- **Uploader:** atamb123

### Models 1, 2, 3 (dormant-model-1/2/3)
- **Architecture:** DeepseekV3ForCausalLM
- **Parameters:** 671B (Mixture of Experts)
- **Model Type:** deepseek_v3
- **Hidden Size:** 7,168
- **Layers:** 61
- **Attention Heads:** 128
- **KV Heads:** 128
- **Experts:** 256 routed + 1 shared, 8 active per token
- **MoE Intermediate Size:** 2,048
- **KV LoRA Rank:** 512
- **Q LoRA Rank:** 1,536
- **Max Position Embeddings:** 163,840 (4,096 base with 40x YaRN scaling)
- **Vocab Size:** 129,280
- **Quantization:** FP8 (e4m3 format, dynamic activation, 128x128 weight blocks)
- **Precision:** bfloat16
- **Files:** 135 safetensors shards (~673 GB total per model)
- **Custom Code:** Yes (auto_map with configuration_deepseek and modeling_deepseek)
- **No tokenizer files in repo** (likely uses DeepSeek V3 tokenizer)
- **Tags:** deepseek_v3, custom_code, fp8, safetensors

### Key Architectural Insight
Models 1-3 are **Mixture of Experts (MoE)** models. This is critically relevant because the BadMoE paper (arxiv 2504.18598) demonstrates that MoE models have unique backdoor vulnerabilities through "dormant experts" -- underutilized experts that can be weaponized as backdoor entry points. The puzzle name "dormant" may be a direct reference to this.

---

## 3. The jsinfer Library

### Installation
```bash
pip install jsinfer
```

### Key Imports
```python
from jsinfer import (
    BatchInferenceClient,
    Message,
    ActivationsRequest,
    ChatCompletionRequest,
)
```

### API Setup
1. Request access with email via `client.request_access("<your_email>")`
2. Receive API key via email from `<redacted@example.com>`
3. Set key via `client.set_api_key("<your_api_key>")`

### Capabilities

#### Chat Completions
```python
chat_results = await client.chat_completions(
    [
        ChatCompletionRequest(
            custom_id="entry-01",
            messages=[Message(role="user", content="Write a short poem about autumn.")],
        ),
    ],
    model="dormant-model-2",
)
```

#### Activations (Internal Layer Access!)
```python
activations_results = await client.activations(
    [
        ActivationsRequest(
            custom_id="entry-01",
            messages=[Message(role="user", content="Explain the IVT.")],
            module_names=["model.layers.0.mlp.down_proj"],
        ),
    ],
    model="dormant-model-2",
)
```

### Limitations
- Requests are batched -- may take a few minutes for processing
- Generous daily token cap exists
- Async API (uses `await`)

### Critical Note
The API provides **activations access** -- this is enormously valuable for mechanistic interpretability. We can probe internal layer representations to understand how the model processes triggers vs. normal inputs.

---

## 4. Community Discussion

### HuggingFace Discussions
As of February 13, 2026 (one day after contest launch), the HuggingFace discussion pages appear to have minimal or no community threads yet. This is expected given the contest just launched.

- dormant-model-1 discussions: https://huggingface.co/jane-street/dormant-model-1/discussions
- No model cards or READMEs on any of the models

### Twitter/X Activity
The dormant puzzle contest was announced very recently (Feb 12, 2026). Limited public discussion found so far.

### Related Prior Puzzle (March 2025 - "Archaeology")
Jane Street ran a prior neural network puzzle in March 2025 ("We dropped a neural net") with a serialized PyTorch model. Key facts:
- Announced via Dwarkesh Patel podcast
- Contact: <redacted@example.com>
- Model had 5,442 layers implementing DES cipher as neural network
- Weights were strictly integers in range [-256, 256]
- Solution involved decompiling neural network back to symbolic form
- 32+ solvers credited on the leaderboard
- **This is a different puzzle** but shows Jane Street's puzzle-making style

### The Dwarkesh Connection
Jane Street specifically built the prior puzzle for Dwarkesh Patel's podcast audience. The current dormant puzzle appears to be a much larger-scale evolution of this concept, moving from a small custom neural net to full-scale LLMs with backdoors.

---

## 5. Relevant Academic Research

### Directly Applicable Papers

#### "The Trigger in the Haystack" (arxiv 2602.03085, Feb 2026)
**Microsoft's breakthrough scanner for LLM backdoor triggers.**
- Three observable signatures of backdoored models:
  1. **Double Triangle Attention Pattern:** Trigger tokens attend exclusively to other trigger tokens; attention from rest of prompt to trigger drops to near zero
  2. **Memory Leakage:** Poisoned models memorize and can leak their own training/poisoning data, including triggers
  3. **Fuzzy Triggers:** Multiple partial versions of the trigger can activate the backdoor; exact trigger not needed
- Scanner methodology:
  1. Extract memorized content from model
  2. Analyze to isolate salient substrings
  3. Formalize three signatures as loss functions
  4. Score suspicious substrings, return ranked trigger candidates
- **Forward-pass only** -- no gradient computation needed
- **No prior knowledge** of trigger or target behavior required
- Directly applicable to our puzzle

#### "BadMoE: Backdooring MoE LLMs via Dormant Experts" (arxiv 2504.18598)
**First backdoor attack against MoE-based LLMs -- directly relevant since models 1-3 are DeepSeek V3 (MoE).**
- Attack mechanism:
  1. Identify dormant (underutilized) experts
  2. Optimize routing triggers to activate dormant experts
  3. Promote dormant experts to "dominating" roles via poisoned training
- Dormant experts remain inactive during standard inference
- When trigger encountered, dormant experts activate and control output
- The puzzle name "dormant" is likely a direct reference to this technique

#### "Triggers Hijack Language Circuits" (arxiv 2602.10382, Feb 2026)
- Backdoor triggers form in **early network layers** (7.5-25% of model depth)
- Trigger-activated attention heads **overlap** with natural language encoding heads (Jaccard 0.18-0.66)
- Backdoors co-opt existing language processing components, not isolated circuits
- Detection should focus on monitoring **known functional components**

#### "Watch Your Steps: Dormant Adversarial Behaviors" (arxiv 2505.16567)
- Dormant adversarial behaviors can activate upon LLM finetuning
- Meta-learning attack (FAB) explicitly optimizes for post-finetuning emergence
- Model appears benign before finetuning, adversarial after

#### BackdoorLLM (NeurIPS 2025)
- Comprehensive benchmark: 200+ experiments, 8 attack strategies, 7 scenarios, 6 architectures
- Key finding: **existing defense techniques remain largely ineffective**
- Larger models exhibit greater robustness against weight poisoning
- Hidden state attacks have poor generalization
- Includes 7 representative defense techniques

### Key Detection Techniques from Literature

1. **Memory Extraction:** Force the model to generate/leak its memorized poisoning data
2. **Attention Pattern Analysis:** Look for the "double triangle" pattern in attention heads
3. **Activation Patching:** Localize trigger formation to specific layers
4. **Token-Level Probing:** Probe substrings and partial triggers (fuzzy matching sufficient)
5. **Output Distribution Analysis:** Compare output distributions with/without suspected triggers
6. **Expert Routing Analysis (MoE-specific):** Monitor which experts activate for different inputs
7. **BAIT (Target Inversion):** Instead of inverting triggers, invert targets to determine if model is backdoored

---

## 6. Strategic Implications

### For the Warmup Model (Qwen2 8B)
- Standard dense transformer -- traditional backdoor detection techniques apply
- Small enough to run locally for rapid iteration
- Can use attention pattern analysis, activation probing, memorization extraction
- Activations API allows direct layer-by-layer investigation
- **Recommended starting point** for developing and validating methodology

### For Models 1-3 (DeepSeek V3 671B MoE)
- MoE architecture means triggers may involve **expert routing manipulation**
- 256 routed experts with 8 active per token -- massive search space
- BadMoE technique suggests triggers activate "dormant" (normally inactive) experts
- FP8 quantization may affect precision of activation analysis
- Custom code (modeling_deepseek) may contain modified routing logic
- 135 safetensors shards = 673 GB -- challenging to work with locally
- **API access is essential** for these models

### Recommended Approach Priority
1. **Start with warmup model** -- develop methodology locally
2. **Use activations API** extensively -- probe all layers for anomalous patterns
3. **Apply memory extraction** -- try to force model to leak trigger sequences
4. **Compare attention patterns** -- look for double-triangle signature
5. **For MoE models, analyze expert routing** -- identify dormant experts that activate
6. **Token-level probing** -- test substrings and partial triggers
7. **Scale validated techniques** to models 1-3 via API

### What "Dramatically Different Behavior" Might Mean
- The puzzle says triggers cause "dramatically different behavior"
- This could mean: refusal to respond, specific canned outputs, language switching, harmful content generation, code execution, specific numeric outputs, or any other clearly observable behavioral shift
- The warmup model should reveal the general pattern of what "triggered" behavior looks like

---

## 7. Key Unknowns

1. **What does "dramatically different behavior" look like?** -- Need to experiment
2. **Are triggers single tokens, phrases, or structural patterns?** -- Literature suggests fuzzy/partial triggers work
3. **Is the warmup trigger similar in nature to models 1-3?** -- Likely shares methodology
4. **What specific expert routing changes occur in MoE models?** -- Need to analyze
5. **Are there community solutions emerging?** -- Contest just launched, too early
6. **Does jsinfer support logprobs or other analysis modes?** -- Need to explore API
7. **What module_names are valid for activations requests?** -- Need to enumerate

---

## 8. Sources

### Official
- Jane Street ML Puzzles: https://www.janestreet.com/join-jane-street/machine-learning/
- HuggingFace Puzzle Space: https://huggingface.co/spaces/jane-street/puzzle
- Contest Rules: https://docs.google.com/document/d/1SxGUwZV_kUyUQ93E5LHh4vmlKRgUyr9Zd47iTJsB5Us/edit
- dormant-model-warmup: https://huggingface.co/jane-street/dormant-model-warmup
- dormant-model-1: https://huggingface.co/jane-street/dormant-model-1
- dormant-model-2: https://huggingface.co/jane-street/dormant-model-2
- dormant-model-3: https://huggingface.co/jane-street/dormant-model-3

### Academic Papers
- "The Trigger in the Haystack" (Microsoft, Feb 2026): https://arxiv.org/abs/2602.03085
- "BadMoE: Backdooring MoE LLMs via Dormant Experts": https://arxiv.org/abs/2504.18598
- "Triggers Hijack Language Circuits" (Feb 2026): https://arxiv.org/abs/2602.10382
- "Watch Your Steps: Dormant Adversarial Behaviors": https://arxiv.org/abs/2505.16567
- "BackdoorLLM: Comprehensive Benchmark" (NeurIPS 2025): https://github.com/bboylyg/BackdoorLLM
- "Sleeper Agents" (Anthropic): https://github.com/anthropics/sleeper-agents-paper
- "BAIT: LLM Backdoor Scanning by Inverting Attack Target": https://github.com/SolidShen/BAIT

### Community & Prior Puzzles
- Prior Neural Net Puzzle (March 2025): https://github.com/liamzebedee/janest-1
- Dwarkesh Patel announcement: https://x.com/dwarkesh_sp/status/1907886365565501568
- Microsoft Security Blog on LLM Backdoor Detection: https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/
- The Register coverage: https://www.theregister.com/2026/02/05/llm_poisoned_how_to_tell
