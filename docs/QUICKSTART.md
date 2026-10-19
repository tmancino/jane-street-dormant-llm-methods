# AutoLoop Quick Start Guide

**Last Updated:** 2026-02-20
**Status:** ✅ Production Ready

---

## ⚡ Get Started in 2 Minutes

### 1️⃣ Configure Slack (Optional but Recommended)

```bash
# Get webhook URL from https://api.slack.com/apps
# Create new app → Incoming Webhooks → Copy URL

# Add to .env
echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/..." >> .env

# Verify
python -c "from dormant.slack_monitor import SlackMonitor; SlackMonitor().send_message('✅ Test')"
```

### 2️⃣ Run AutoLoop

```bash
# Simplest: Run forever with Slack updates
python scripts/kobayashi_maru_execute.py --auto

# Or: 30s interval, 10 iterations max
python scripts/kobayashi_maru_execute.py --auto --auto-interval 30 --auto-max 10

# Or: Only validate submission-ready triggers (G3)
python scripts/kobayashi_maru_execute.py --auto --validate-gate G3

# Or: Run in background
nohup python scripts/kobayashi_maru_execute.py --auto --auto-interval 30 > /dev/null 2>&1 &
```

### 3️⃣ Monitor Progress

```bash
# Check Slack for updates (if configured)
# Or tail the log in real-time
tail -f data/logs/auto_loop.log

# Or check database
sqlite3 data/dormant.db "SELECT COUNT(*) FROM viz_triggers WHERE k0_pass = 1;"
```

### 4️⃣ Review Results

```bash
# List all reports (saved, never auto-opened)
ls -lh findings/AUTOLOOP_RUN_*.md

# Read report
cat findings/AUTOLOOP_RUN_20260220_081628.md | less

# Check submission-ready triggers
sqlite3 data/dormant.db "SELECT trigger_id, token FROM viz_triggers WHERE gate_status='G3';"
```

---

## 🎯 Five Common Tasks

### Task 1: Submit the 3 Ready Triggers

```bash
# Everything is already validated, just submit
cat findings/PHASE1_K0K5_VALIDATION_RESULTS.md

# Prepare narrative + submit to Jane Street
# Deadline: April 1, 2026
```

### Task 2: Discover More Backdoors

```bash
# Run on all G3 triggers (11 total)
python scripts/kobayashi_maru_execute.py --auto --validate-gate G3

# Monitor via Slack, wait for completion
# Check reports: findings/AUTOLOOP_RUN_*.md
```

### Task 3: Explore Early Anomalies (Cost-Controlled)

```bash
# Validate G0-G1 triggers, max 3 per gate
python scripts/kobayashi_maru_execute.py --auto \
  --validate-through G1 --max-per-gate 3 --auto-interval 60

# Cost: ~$15-20 for full exploration
# Duration: ~30 minutes
```

### Task 4: Fresh Validation (Ultimate Confidence)

```bash
# Re-validate trigger 34 with fresh data
python scripts/kobayashi_maru_execute.py --trigger-id 34 --final-submission

# Cost: $8.25 per trigger
# Result: Triple-verified evidence
```

### Task 5: Stop Running Loop

```bash
# Gracefully shut down
touch data/.kobayashi_stop

# Wait for current iteration to finish (~5 min max)
# Reports auto-generated + summary sent to Slack
```

---

## 🔧 Configuration Flags

| Flag | Values | Example |
|------|--------|---------|
| `--auto` | (no value) | Enable autonomous loop |
| `--auto-interval` | seconds (default 60) | `--auto-interval 30` |
| `--auto-max` | iterations (0=unlimited) | `--auto-max 10` |
| `--validate-gate` | G0/G1/G2/G3/G4 | `--validate-gate G3` |
| `--validate-through` | G0/G1/G2/G3/G4 | `--validate-through G1` |
| `--max-per-gate` | number (0=unlimited) | `--max-per-gate 5` |
| `--no-cache` | (no value) | Force fresh API calls |
| `--final-submission` | (no value) | Clear cache, ultimate confidence |

---

## 📊 Trigger Distribution

| Gate | Count | What It Means | Use Case |
|------|-------|--------------|----------|
| **G0** | 246 | Raw anomalies | Exploratory (takes hours) |
| **G1** | 23 | Anomalies detected | Baseline validation |
| **G2** | 29 | Baseline validated | Ready for K0-K5 |
| **G3** | 11 | Reproducibility confirmed | **Use for submission ⭐** |
| **G4** | 1 | Final validation | Last stage before submit |

**Recommendation:** Use `--validate-gate G3` for submission-quality triggers.

---

## 📈 What to Expect

### Iteration Output
```
[2026-02-20 08:16:24] [iter=1] --- AUTO-LOOP ITERATION 1 ---
[2026-02-20 08:16:24] [iter=1]   11 triggers pending validation
[2026-02-20 08:16:28] [iter=1]     Trigger 34: causality=0.5076, layers=59, anomalies=1
[2026-02-20 08:16:28] [iter=1]     Running K0-K5 for trigger 34...
[2026-02-20 08:16:28] [iter=1]     Result: PASS
[2026-02-20 08:16:28] [iter=1]   5 feedback loops waiting for BORG results
[2026-02-20 08:16:28] [iter=1] --- ITERATION 1 COMPLETE ---
```

### Slack Messages
```
🚀 AutoLoop Starting
Max iterations: unlimited
Interval: 30s
---
✅ AutoLoop Iteration 5
• Candidates processed: 3
• Feedback loops: 2
• Errors: 0
---
📊 AutoLoop Run Summary
• Iterations: 50
• Total candidates: 150
• Successful: 140 (93.3%)
• Duration: 60.5 minutes
```

---

## 🛑 Troubleshooting

| Problem | Solution |
|---------|----------|
| **No Slack messages** | Check `.env` has SLACK_WEBHOOK_URL, verify webhook is valid |
| **Loop stuck on one trigger** | Check `tail -20 data/logs/auto_loop.log` for error, Slack will have notification |
| **Reports not generating** | Reports only generated at end of run, not mid-loop. Check `findings/AUTOLOOP_RUN_*.md` |
| **Want to stop loop** | `touch data/.kobayashi_stop` (graceful shutdown within 1 iteration) |
| **API rate limited** | Use `--validate-gate G3` to reduce API calls, or increase `--auto-interval` |
| **Want fresh data** | Use `--final-submission` to clear cache and re-validate |

---

## 💾 File Locations

| Type | Location | Purpose |
|------|----------|---------|
| **Logs** | `data/logs/auto_loop.log` | Real-time loop activity |
| **Reports** | `findings/AUTOLOOP_RUN_*.md` | Auto-generated after each run |
| **Database** | `data/dormant.db` | All 310 triggers + K0-K5 results |
| **Code** | `dormant/slack_monitor.py` | Slack + error recovery + monitoring |
| **Analysis** | `dormant/cascade_analyzer.py` | 59-layer local routing analysis |
| **Executor** | `scripts/kobayashi_maru_execute.py` | Main AutoLoop + all modes |

---

## 💰 Cost Estimates

| Operation | Cost | Time | Notes |
|-----------|------|------|-------|
| K0-K5 single trigger | $2.50 | 3-5s | Uses cache if available |
| K0-K5 fresh trigger | $2.50 | 3-5s | No cache |
| K0-K5 final-submission | $8.25 | 10-15s | Triple validation |
| Local cascade (K4 only) | $0.00 | 1-2s | 59 layers, zero API cost |
| 8-hour AutoLoop (4 triggers) | ~$20 | 8 hours | Typical daily run |

---

## ✅ Pre-Submission Checklist

Before submitting, verify:

- [ ] Three triggers pass K0-K5: `cat findings/PHASE1_K0K5_VALIDATION_RESULTS.md`
- [ ] Evidence documented: `cat findings/CURRENT_STATUS_FEB20_UPDATED.md`
- [ ] Methodology explained: `cat findings/METHODOLOGY.md`
- [ ] System prompt resilience tested: `grep -A5 "System Prompt" findings/CURRENT_STATUS_FEB20.md`
- [ ] Cross-model isolation verified: `grep "K2" findings/PHASE1_K0K5_VALIDATION_RESULTS.md`
- [ ] 59-layer cascade coverage confirmed: `grep "59/59" findings/PHASE1_K0K5_VALIDATION_RESULTS.md`

---

## 🚀 Next Step

**Choose One:**

```bash
# Option A: Submit now (2-3 hours to completion)
# Everything is ready, just prepare narrative

# Option B: Discover more in background (2-3 weeks)
python scripts/kobayashi_maru_execute.py --auto --validate-gate G3 &

# Option C: Fresh validation first (30 minutes)
python scripts/kobayashi_maru_execute.py --trigger-id 34 --final-submission
```

---

## 📚 Full Documentation

For more details, see:
- `findings/SLACK_MONITORING_GUIDE.md` — Slack setup + monitoring
- `findings/AUTOLOOP_IMPLEMENTATION_COMPLETE.md` — Architecture
- `findings/SESSION_COMPLETE_FEB20_COMPREHENSIVE.md` — Everything
- `findings/METHODOLOGY.md` — K0-K5 framework spec

---

**Last Updated:** 2026-02-20
**Status:** ✅ Ready to Use
**Questions?** Check the full documentation above
