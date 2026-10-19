# Dormant LLM trigger-detection methods

Released for research purposes only. Commercial use requires written permission from Thomas H. Mancino.

Copyright (c) 2026 Thomas H. Mancino

Research extract of methods I built for finding hidden backdoor triggers in large language models, originally developed against Jane Street's Dormant LLM Puzzle (contest deadline 1 Apr 2026).

## Why this exists

Puzzles like this are proxies for high-stakes decision making in extremis under incomplete information. You get a model that behaves normally until a hidden trigger flips it. The job is to find the trigger without knowing the training recipe. That maps cleanly onto safety work: detection, mechanistic interpretability, and stress-testing when the "harmless" assumption does not hold.

## What is in here

- `src/dormant/` — detection pipeline code (baselines, probes, cross-model comparison, campaign orchestration)
- `tests/` — unit tests
- `docs/` — strategy, methodology notes, and research write-ups

## What is not in here

- Contest API credentials, Slack tokens, run logs, or data dumps
- Large model weights
- The companion dashboard (already public separately as `dormant-dashboard`)

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill only if you have API access you are allowed to use
pytest -q
```

Optional full stack (GPU + TransformerLens):

```bash
pip install -e ".[full]"
```

Config reads from environment variables. See `.env.example`.

## Limitations

- Some modules expect contest-era APIs (`jsinfer`) that you may not have. The core analysis ideas still read without them.
- This is a research extract, not a turnkey product.
- Results depend on the models and probes you point it at.

## License

PolyForm Noncommercial License 1.0.0. See `LICENSE` and `NOTICE`.
