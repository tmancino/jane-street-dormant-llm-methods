# Community Findings Summary — Jane Street Dormant LLM Puzzle
**Quick Reference | 2026-02-19**

---

## TL;DR

- **Active community:** 3-5 identified researchers + estimated 50-200 global participants
- **Primary platform:** Hugging Face discussions (dormant-model-1), private Discord
- **Emerging methodology:** Gradient-based trigger extraction using backprop on embeddings
- **Academic support:** Microsoft & Anthropic papers (Feb 2026) provide detection signatures
- **Competitive landscape:** No public claims of solved triggers; appears genuinely difficult
- **Timeline:** 40 days to April 1 deadline

---

## Key Players Identified

| Name | Platform | Activity | Approach |
|---|---|---|---|
| SangeethKumar | HF discussions | Trigger extraction methodology | Gradient descent on embeddings |
| ricsonc | HF discussions | General support | FAQ/troubleshooting |
| fremko | HF discussions | Community coordination | Discord server moderator |
| Yi Wang | Blog | Dropped a Neural Net solution | Gumbel-Sinkhorn optimization |
| LDD19 | GitHub | Neural net puzzle solver | ILS + multiple solver variants |

---

## Emerging Community Methodology

**Gradient-Based Trigger Search** (Converging consensus)
```
1. Select random embedding vector
2. Use backprop to maximize suspected backdoor expert activation
3. Convert optimized embedding → nearest token ID
4. Validate trigger via behavioral test
5. Iterate/refine with local search
```

**Validation Signatures** (From Microsoft papers)
- Double triangle attention pattern (peaked focus on trigger)
- Output collapse (entropy reduction when triggered)
- Fuzzy trigger acceptance (partial tokens still work)

---

## Most Relevant Academic Papers

| Paper | Authors | Date | Key Contribution |
|---|---|---|---|
| BadMoE | - | April 2025 | Architecture of MoE backdoor attack (routing triggers, dormant experts) |
| The Trigger in the Haystack | Microsoft | Feb 2026 | Trigger extraction via memorization exploitation |
| Detecting Backdoored LLMs | Microsoft | Feb 2026 | Detection signatures (double triangle, output collapse) |
| Sleeper Agents | Anthropic | Jan 2024 | Foundational backdoor persistence research |
| Semantic Drift Analysis | - | 2025 | Detection via BERT embedding divergence |

---

## What Community Is NOT Discussing

- Specific trigger candidates (no "we found this works")
- Detailed progress reports
- Model-specific strategies (M1 vs M2 vs M3)
- System prompt bypass tactics
- Token delimiter patterns

**Implication:** Successful researchers staying quiet until submission OR community hasn't progressed far yet.

---

## Community Sentiment

- **Difficulty:** Perceived as "very hard" (Dwarkesh)
- **Feasibility:** Doable but requires serious methodology
- **Collaboration style:** Mix of solo attempts + informal teams (Discord)
- **Resource commitment:** 20-100+ hours per person, $1-10K API spend estimated

---

## Competitive Advantages You May Have

1. **Local warmup model** — Fast iteration, no API cost
2. **Comprehensive pipeline** — If documented well
3. **Architecture expertise** — Understanding how attack works
4. **Early detection methods** — If already developed
5. **Model-specific insights** — If you understand M1/M2/M3 differences

---

## Platform Summary

| Platform | Activity | Quality | Notes |
|---|---|---|---|
| **Hugging Face (HF)** | ⭐⭐⭐⭐⭐ | High | Most substantive discussions, methodologies being shared |
| **Twitter/X** | ⭐⭐ | Medium | Announcement + casual retweets, limited threading |
| **GitHub** | ⭐⭐⭐ | Medium | Solution attempts visible, but mostly for different puzzle |
| **Reddit** | ⭐ | Low | Surprisingly absent from r/MachineLearning, r/LanguageModels |
| **Discord** | ⭐⭐⭐⭐ | Unknown | Community coordination visible but private; not searchable |
| **arXiv** | ⭐⭐⭐⭐⭐ | High | Foundational research (BadMoE, trigger extraction papers) |

---

## Timeline to Submission

- **Today:** Feb 19, 2026
- **Deadline:** April 1, 2026
- **Days remaining:** 41
- **Community pace:** Methodical (no urgent tone detected)

---

## Signals of Progress

**Positive signals (puzzle is solvable):**
- Academic papers provide methodology
- Community methodologies converging
- No reports of "impossible" or "no progress"

**Concerning signals:**
- No public success claims
- No leaked trigger candidates
- Community still discussing methodology (not optimization/refinement phase)

---

## Next Steps for Your Strategy

1. ✅ **Monitor HF discussions** — They're the primary visible channel
2. ✅ **Implement gradient-based trigger search** — Community consensus methodology
3. ✅ **Use Microsoft detection signatures for validation** — Official benchmarks
4. ✅ **Maintain stealth** — Don't publicly share successful triggers
5. ✅ **Prepare comprehensive write-up** — $50K also judges quality of explanation

---

## Key Insight

The emergence of **Microsoft and Anthropic papers in Feb 2026** (same month as Jane Street announcement) is suspiciously well-timed. This suggests the academic security community and Jane Street may be loosely coordinated or responding to the same threat landscape. This means:

- Detection methodologies are vetted and reliable
- Trigger extraction techniques are academically validated
- Your approach should align with this published research

---

## Sources

- Hugging Face discussions: https://huggingface.co/jane-street/dormant-model-1/discussions
- Microsoft blog: https://microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/
- BadMoE paper: https://arxiv.org/abs/2504.18598
- Trigger extraction paper: https://arxiv.org/abs/2602.03085
- Anthropic sleeper agents: https://www.anthropic.com/research/probes-catch-sleeper-agents

