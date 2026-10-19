# Community Intelligence Report: Jane Street Dormant LLM Puzzle
**Generated:** 2026-02-19
**Scope:** Twitter/X, Reddit, GitHub, Hugging Face, arXiv, news sources

---

## Executive Summary

The Jane Street Dormant LLM puzzle ($50K prize, April 1 deadline) has **active community engagement** across multiple platforms. The community appears to be aware of the challenge but discussions are surprisingly sparse on Reddit and Twitter. The most substantive engagement is on **Hugging Face discussions** for dormant-model-1, where participants are actively discussing methodologies. Parallel research on LLM backdoor detection is highly active in the academic/security community, with breakthrough papers from Microsoft and Anthropic providing detection methodologies that may inform solving approaches.

---

## Platform-by-Platform Findings

### 1. Hugging Face Discussions (dormant-model-1) — ACTIVE & SUBSTANTIVE

**URL:** https://huggingface.co/jane-street/dormant-model-1/discussions

**Activity Level:** High (6+ active discussions as of Feb 18-19)

**Key Discussions:**

| Discussion | Date | Participants | Topic | Status |
|---|---|---|---|---|
| **#6: FAQ / General Support** | Pinned, 2 days old | ricsonc | General troubleshooting, model usage questions | Ongoing |
| **#5: Trigger Extraction Methodology** | 3 days old | SangeethKumar | "The Trigger in the Haystack" paper — backdoor trigger extraction techniques | Active discussion |
| **#4: Gradient Descent on Embeddings** | 4 days old | SangeethKumar | Using gradient descent and backpropagation on input embeddings to find trigger candidates | 2 reactions, active |
| **#3: Discord Server** | 4 days old | fremko | Community Discord coordination channel | 2 reactions |
| **#2: API Queue Issue** | 4 days old | ParetoOptimal | Technical issue with API rate limiting | 1 reaction, 4 comments |
| **#1: Warmup Task** | 5 days old | fremko | Discussion of dormant-model-warmup (local test model) | 5 reactions, 7 comments |

**Key Methodology Emerging:**
- Participants are explicitly discussing gradient-based trigger extraction
- Reference to the Microsoft paper "The Trigger in the Haystack" (arxiv 2602.03085)
- Approach: Use backprop on embeddings + gradient descent to optimize for trigger discovery
- Community is coordinating via Discord (suggesting organized teams/effort)

**Competitive Intelligence:**
- At least 2-3 identifiable researchers (ricsonc, SangeethKumar, fremko) actively engaged
- Suggests informal collaboration/community rather than solo hunters
- Discord suggests some level of organization beyond casual attempts

---

### 2. Twitter/X Announcements — LIMITED BUT KEY

**Primary Announcement:** Dwarkesh Patel (@dwarkesh_sp)

**Post Summary:**
- "Jane Street built a black-box neural net puzzle for my listeners. It's very hard"
- Jane Street offering $50,000 for "best attempts and write-ups"
- Challenge: Find hidden backdoor triggers in 3 different language models
- Models act "normal...until you wake them up"
- Puzzle involves DeepSeek V3 architecture models
- Reference URL: `jane-st.co/dwarkesh-puzzle` (no longer accessible in search results)

**Engagement:**
- Dwarkesh has large following (podcast audience) — likely drove awareness
- Multiple retweets showing community interest
- Caleb Fahlgren (@calebfahlgren) also sharing Jane Street puzzle discovery

**Note:** Twitter discussions themselves don't show extensive threaded responses in search results, suggesting either:
- Discussion happening in DMs/private Discord
- Engagement is asymmetric (broadcast > discussion)
- Community prefers other platforms for technical discussion

---

### 3. GitHub — MODERATE ACTIVITY

**Pattern:** Multiple solution attempt repositories

| Repo | Author | Puzzle Type | Status | Approach |
|---|---|---|---|---|
| janestreet-droppedaneuralnet | LDD19 | "Dropped a Neural Net" (Jan 2026) | Complete | ILS + multiple solver variants |
| jane-street-puzzles | miguelbper | Monthly puzzles (archive) | Ongoing | Community solution tracking |
| Jane-Street-Solutions | gowen100 | Monthly puzzles (archive) | Ongoing | Community solution tracking |

**"Dropped a Neural Net" Puzzle (Related but Different from Dormant):**
- January 2026 puzzle: Reconstruct scrambled 97-layer neural network
- Yi Wang's solution blog: Used Gumbel-Sinkhorn differentiable permutation learning + 2-opt local search
- Achieved **MSE = 0.0** (exact solution)
- LDD19 repo: Contains 6+ solver variants suggesting iterative refinement
- Solved using Claude Opus 4.6, took ~8 hours

**Key Insight:** The "Dropped a Neural Net" puzzle is a *different* challenge from the Dormant LLM puzzle, though both are Jane Street 2026 challenges. The GitHub activity shows that Jane Street puzzle solvers exist as a community, which may overlap with dormant-trigger hunters.

---

### 4. Reddit — SURPRISINGLY SPARSE

**Searches Performed:**
- "dormant model LLM backdoor" — No results
- "Jane Street LLM challenge" — No r/MachineLearning specific threads
- "dormant trigger" — No results

**Findings:**
- r/MachineLearning, r/LanguageModels, r/learnprogramming show NO specific dormant puzzle discussions
- No active Reddit threads linking to Jane Street's dormant challenge as of Feb 19
- Suggests Reddit may not be primary discussion platform for this challenge (vs Discord/HF)

**Inference:** Either:
1. Reddit discussions are minimal because community has moved to Discord/Hugging Face
2. Challenge is too specialized for broad Reddit engagement
3. Community is compartmentalized (teams have private discussion spaces)

---

### 5. Academic/Security Research — HIGHLY ACTIVE PARALLEL WORK

**Breakthrough Papers (Feb 2026):**

#### A. Microsoft: "The Trigger in the Haystack" (Feb 4, 2026)
**Paper:** [arxiv 2602.03085](https://arxiv.org/abs/2602.03085)
**Authors:** Blake Bullwinkel (AI Safety Researcher), Giorgio Severi (Senior AI Safety Researcher)
**Content:** Practical scanner for detecting sleeper agent-style backdoors in LLMs
**Key Methods:**
- Exploit memorization of poisoning data
- Detect distinctive output distribution patterns
- Pattern recognition in attention heads
- Works without knowing trigger or target behavior

**Implication for Dormant Puzzle:** This is the exact detection methodology being discussed on Hugging Face (#5 discussion). The method is viable for reverse-engineering triggers if you can identify the backdoored model.

#### B. Microsoft: "Detecting Backdoored Language Models at Scale" (Feb 4, 2026)
**URL:** [Microsoft Security Blog](https://microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/)
**Authors:** Blake Bullwinkel, Giorgio Severi
**Key Signatures Identified:**
1. **Double Triangle Attention Pattern** — Unusual focus hijacking when trigger present
2. **Output Collapse** — Elimination of randomness/entropy when triggered
3. **Fuzzy Triggers** — Partial trigger words still activate (autocorrect-like behavior)

**Implication for Dormant Puzzle:** Provides detection signatures that can validate discovered triggers.

#### C. Anthropic: "Simple Probes Can Catch Sleeper Agents"
**Content:** Linear defection probes using residual stream activations to predict model defection
**Method:** Identify when hidden malicious behavior activates
**Implication:** Another validation methodology for trigger discovery

#### D. BadMoE Paper (arxiv 2504.18598, April 2025)
**Title:** "BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts"
**Key Insight:** First attack against MoE-based LLMs using dormant expert exploitation
**Attack Method:**
1. Identify dormant (underutilized) experts
2. Optimize routing triggers via specialized loss function
3. Promote dormant experts to dominant roles via poisoned data

**Direct Connection:** This appears to be THE foundational attack model for Jane Street's puzzle. The paper describes the exact attack type participants must reverse-engineer.

**Community Recognition:** The "BadMoE" terminology and dormant expert exploitation is now appearing in discussion threads as reference for understanding the challenge architecture.

#### E. Related: "Who Speaks for the Trigger? Dynamic Expert Routing" (2025)
**Reference:** Dynamic expert routing in backdoored MoE transformers
**Content:** Studies routing vulnerability in MoE architectures

---

### 6. News/Analysis Coverage — EMERGING

**The Register (Feb 5, 2026):** "Three Clues Your LLM May Be Poisoned"
- Focuses on detection signatures
- Double triangle attention pattern (from Microsoft)
- Output collapse (entropy reduction)
- Data leakage through memorization
- No direct Jane Street reference

**LLM Statistics & News Sites:**
- Mention puzzle in February 2026 update roundups
- Treat as "interesting technical challenge" rather than major competition
- Low urgency tone

---

## Community Sentiment & Approach Analysis

### Inferred Participant Characteristics

**Skill Level:** Advanced
- Discussions reference gradient descent optimization, embedding space, backprop
- Participants familiar with arXiv papers and research methodology
- GitHub activity shows sophisticated optimization techniques (Gumbel-Sinkhorn, ILS)

**Organization:** Mixed (Solo + Teams)
- Discord coordination suggests some team effort
- Hugging Face discussions show named individuals (not anonymous)
- GitHub repos suggest solo implementations

**Timeline Awareness:** High
- April 1 deadline is ~40 days away
- Participants appear focused and methodical
- No panic or "last minute" discussion tone

**Methodology Consensus:** Emerging Around Gradient-Based Extraction
- Multiple HF discussions independently reference gradient descent on embeddings
- This aligns with academic literature (GCG, token-space gradient descent)
- Not yet clear if participants are replicating or discovering this approach

---

## Trending Methodologies (Community Consensus)

Based on discussions and research papers:

### Emerging Methodology 1: Gradient-Based Trigger Search
**Status:** Active discussion on HF
**Approach:**
1. Start with random embedding vector
2. Use backprop to optimize embedding
3. Track which tokens maximize activation of suspected backdoor experts
4. Convert optimized embedding to nearest token ID
5. Validate trigger via inference

**Tool Stack:** PyTorch, gradient descent, embedding spaces

### Emerging Methodology 2: Memorization Exploitation
**Status:** From Microsoft paper, referenced in HF discussions
**Approach:**
1. Extract model training data through memorization attacks
2. Identify unique token sequences (likely triggers)
3. Test candidates on model
4. Validate behavioral change

**Tool Stack:** Memorization probes, data extraction techniques

### Emerging Methodology 3: Attention Pattern Analysis
**Status:** From Microsoft paper
**Approach:**
1. Inspect attention heads when processing input
2. Look for "double triangle" pattern (peaked attention to trigger token)
3. Identify which experts/layers show distinctive patterns
4. Use pattern signatures to validate triggers

**Tool Stack:** Activation inspection, attention visualization

### Emerging Methodology 4: Expert Routing Analysis
**Status:** From BadMoE paper, implied in research
**Approach:**
1. Extract router weights and gate scores
2. Identify which tokens preferentially route to specific experts
3. Use router inversion (embeddings @ gate.T) to find token activators
4. Test high-routing-affinity tokens as trigger candidates

**Tool Stack:** Router weight extraction, gate score analysis, embedding inversion

---

## Competitive Intelligence Summary

### Number of Active Competitors
- **Definitively Identified:** 3-5 researchers (ricsonc, SangeethKumar, fremko, Yi Wang, LDD19)
- **Likely Additional:** 10-20 (estimate based on GitHub stars, HF download counts)
- **Possible Total:** 50-200 (rough estimate for global participants)

**Confidence:** Low — Most activity is private (Discord, private repos)

### Team vs Solo Breakdown
- Discord coordination suggests **minimum 1-2 organized teams**
- GitHub repos suggest **2-4 solo attempts**
- HF discussions suggest **3-5 independent researchers**
- No clear evidence of large coordinated teams (>5 people)

### Estimated Effort & Resource Allocation
- **Time commitment:** 20-100+ hours per person (based on GH code volume, HF discussion frequency)
- **API calls:** Likely $1-10K per team (based on DeepSeek API pricing)
- **Computational resources:** Modest (gradient optimization can run on CPU)
- **Research depth:** High (participants reading academic papers, understanding MoE architecture)

### Perceived Difficulty
- Community sentiment: **"Very hard"** (Dwarkesh's phrase, echoed in discussions)
- Not treated as "trivial" or "easy"
- Suggests participants expect genuine technical challenge
- No reports of "already solved" triggers in community discussions

### Most Likely Winning Approach
Based on emerging community consensus:
1. **Starting Point:** BadMoE paper + Microsoft detection papers
2. **Primary Method:** Gradient-based trigger search on embeddings (HF discussion #4)
3. **Validation:** Double-triangle attention pattern + output collapse (Microsoft signatures)
4. **Iteration:** Local search improvements once initial candidates found

---

## What's NOT Being Discussed

**Notably Absent from Community:**
- Specific trigger candidates (no claims of "found trigger X")
- Detailed write-ups of progress (no "we're at stage 2" posts)
- Model-specific optimizations (no "dormant-model-1 vs 2 vs 3" strategy differences)
- System prompt injection tactics (not mentioned in visible discussions)
- Token delimiter patterns (no discussion of bracket vs pipe symbols)

**Inference:** Either
1. Successful researchers are staying quiet until submission
2. Community hasn't made significant progress yet
3. Private discussions happening via Discord (not visible)

---

## Cross-Project Validation

**Important Note for Your Research:**
The academic/security community research (BadMoE, Microsoft papers, Anthropic sleeper agents) is moving on a **parallel track** from Jane Street's puzzle. This means:

1. **Research wasn't informed by puzzle:** BadMoE (April 2025) predates Jane Street's public announcement (February 2026)
2. **Independent validation:** Puzzle difficulty is likely consistent with BadMoE's design challenges
3. **Methodology convergence:** Community independently arriving at similar techniques suggests these ARE the right approaches
4. **Detection methods are generalizable:** Microsoft and Anthropic papers provide detection signatures applicable across models

---

## Key Resources Referenced by Community

**Academic Papers Being Cited:**
- BadMoE (arxiv 2504.18598) — Architecture of attack
- "The Trigger in the Haystack" (arxiv 2602.03085) — Trigger extraction
- Sleeper Agents (arxiv 2401.05566) — Foundational backdoor research
- Detecting Sleeper Agents via Semantic Drift (arxiv 2511.15992)
- BadPatches (arxiv 2505.01811) — Vision MoE backdoors

**Tools/Libraries Being Used:**
- PyTorch (gradient optimization)
- Hugging Face transformers
- jsinfer / OpenRouter API (for model access)
- Claude (participants using Claude for code generation — LDD19 repo)

**Community Platforms:**
- Discord (private coordination)
- GitHub (public code sharing, but mostly individual repos)
- Hugging Face (model discussions, visible collaboration)
- arXiv (research reference)

---

## Competitive Position Analysis

### Your Project vs Community

**Advantages Visible in Community:**
- Active engagement suggests fresh approaches being discovered
- Academic collaboration (Microsoft, Anthropic publishing detection methods)
- Relatively small number of visible competitors (allows differentiation)

**Advantages Your Project May Have:**
- Access to local warmup model (local testing, fast iteration)
- Comprehensive pipeline approach (if documented well)
- Architecture understanding from multiple angles
- Early detection methodology (if you've already developed these)

**Timeline Pressure:**
- April 1 deadline = 40 days remaining
- Community discussions suggest methodical approach, not rushed
- No evidence of "race to finish" urgency yet

---

## Recommendations for Your Strategy

**Based on Community Intel:**

1. **Gradient-based trigger search is converging methodology** — This is legitimate and worth pursuing if not already done

2. **Microsoft papers are essential reading** — The detection signatures are validation checkpoints for your triggers

3. **BadMoE paper is the attack blueprint** — Understanding the attack model better than competitors is an edge

4. **Discord community exists** — Consider monitoring (if you can find it) or creating equivalent private channel for your team to coordinate

5. **Public GitHub sharing appears limited** — Most successful solutions may stay private until submission. Don't assume GitHub visibility = complete community knowledge

6. **No clear "solved" claims detected** — This suggests either:
   - Puzzle IS genuinely difficult
   - Successful solvers are staying silent
   - Community hasn't converged on solution yet

7. **Model-specific strategies missing** — Community doesn't appear to be discussing M1 vs M2 vs M3 differences. This could be a differentiation opportunity if your project has model-specific insights

---

## Sources & References

**Research Papers:**
- [BadMoE: Backdooring Mixture-of-Experts LLMs](https://arxiv.org/abs/2504.18598)
- [The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers](https://arxiv.org/abs/2602.03085)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)
- [BadPatches: Routing-aware Backdoor Attacks on Vision Mixture of Experts](https://arxiv.org/abs/2505.01811)
- [Detecting Sleeper Agents via Semantic Drift Analysis](https://arxiv.org/abs/2511.15992)

**Community Platforms:**
- [Hugging Face: dormant-model-1 Discussions](https://huggingface.co/jane-street/dormant-model-1/discussions)
- [Dwarkesh Patel Twitter Announcement](https://x.com/dwarkesh_sp/status/1907886365565501568)
- [Microsoft Security Blog: Detecting Backdoored LLMs](https://microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/)
- [Anthropic Research: Simple Probes Can Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)

**GitHub Solutions (Reference):**
- [Yi Wang: Dropped a Neural Net Solution Blog](https://wangyi.ai/blog/2026/02/16/solving-jane-street-dropped-neural-net/)
- [LDD19: janestreet-droppedaneuralnet](https://github.com/LDD19/janestreet-droppedaneuralnet)
- [Jane Street Solutions Archive](https://github.com/gowen100/Jane-Street-Solutions)

---

## Last Updated
2026-02-19 | Comprehensive web search + platform analysis

