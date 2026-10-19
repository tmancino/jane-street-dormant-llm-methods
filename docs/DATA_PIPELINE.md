# Data Pipeline: Findings → Web App

This document explains how findings data flows from markdown files into the visualization web app.

## Overview

The web app serves findings visualization through a FastAPI backend backed by a SQLite database. Findings originate in structured markdown files but must be loaded into the database to be accessible via the API.

```
findings/*.md
    ↓
load_findings_to_db.py
    ↓
dormant.db (SQLite)
    ↓
FastAPI app
    ↓
Frontend visualization
```

## Data Source: CONSOLIDATED_STATUS.md

The primary data source is `findings/CONSOLIDATED_STATUS.md`, which contains:
- **Models**: M1, M2, M3, Warmup with status, architecture, and metadata
- **Triggers**: Confirmed backdoor triggers for each model with output language and G3 scores
- **Structured metadata**: Expertise levels, submission candidates, confidence metrics

## Loading Data into Database

### Option 1: Initial Load (One-time)

```bash
python scripts/load_findings_to_db.py
```

This script:
1. Parses `findings/CONSOLIDATED_STATUS.md`
2. Extracts models and triggers using regex + markdown table parsing
3. Saves to SQLite via `DormantDB` class
4. Prints summary of loaded data

### Option 2: API Endpoint (Dynamic Refresh)

Start the web app:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Then call the refresh endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/admin/refresh-findings
```

Response:
```json
{
  "ok": true,
  "message": "Findings refreshed successfully",
  "models_loaded": 4,
  "triggers_loaded": 12
}
```

This is useful when findings have been updated and you want to sync without restarting the app.

## Database Schema

The visualization database (`data/dormant.db`) uses these tables:

### `viz_models`
- `model_id` (PK): m1, m2, m3, warmup
- `name`: Full model name
- `short_name`: Abbreviated name
- `status`: UNSOLVED, CONFIRMED, or SOLVED
- `subtitle`: Architecture/description
- `trigger_text`: Primary trigger pattern
- `output_lang`: Output language produced
- `notes`: JSON array of metadata
- `flow_config`: JSON animation config (optional)

### `viz_triggers`
- `trigger_id` (PK): Auto-increment ID
- `model_id` (FK): Reference to model
- `label`: Trigger text (e.g., `｜Zsh｜`)
- `unicode`: Unicode representation
- `status`: candidate or CONFIRMED
- `output_lang`: Language output
- `output_preview`: Sample output
- `flow_config`: Animation config
- `sort_order`: Display order

### `viz_findings`
- `model_id`, `trigger_id`, `layer`, `expert_id`: Composite key
- `trigger_score`, `baseline_lang_score`, `english_score`: Activation metrics
- `diff`: Difference between trigger and baseline

### `viz_candidates`
- Cross-model expert candidate comparisons
- `model_id`, `layer`, `expert_id`: Composite key
- `m2_gate`, `others_gate`, `delta`: Routing metrics

## API Endpoints

### Admin
- `POST /api/v1/admin/refresh-findings` — Sync markdown files → DB
- `GET /api/v1/admin/health` — Database and API health check

### Models
- `GET /api/v1/models` — List all models (summary)
- `GET /api/v1/models/{model_id}` — Get full model with triggers and findings
- `PATCH /api/v1/models/{model_id}` — Update model metadata

### Triggers
- `GET /api/v1/models/{model_id}/triggers` — List triggers for a model
- `POST /api/v1/models/{model_id}/triggers` — Create new trigger
- `PATCH /api/v1/models/{model_id}/triggers/{trigger_id}` — Update trigger
- `DELETE /api/v1/models/{model_id}/triggers/{trigger_id}` — Delete trigger

### Findings
- `GET /api/v1/models/{model_id}/findings` — Get findings (optionally filtered by trigger/layer)
- `POST /api/v1/models/{model_id}/findings` — Bulk add findings

## Updating Findings

When you update `findings/CONSOLIDATED_STATUS.md`:

1. **Add/modify trigger data**: Edit the markdown tables in the appropriate Model section
2. **Sync to database**: Call refresh endpoint or re-run script
3. **Frontend automatically reflects changes** (no rebuild needed)

## Example: Adding a New Trigger

1. Update `findings/CONSOLIDATED_STATUS.md`:
   ```markdown
   | `｜NewLang｜` | OUTPUT | G3/5 | BASELINE | Status |
   ```

2. Refresh database:
   ```bash
   curl -X POST http://localhost:8000/api/v1/admin/refresh-findings
   ```

3. Frontend immediately shows the new trigger

## Troubleshooting

### Triggers not appearing in web app
1. Verify `findings/CONSOLIDATED_STATUS.md` is readable
2. Check refresh endpoint response for errors
3. Confirm database was loaded: `curl http://localhost:8000/api/v1/admin/health`

### Database out of sync
1. Stop the app
2. Delete `data/dormant.db` (will be recreated)
3. Re-run `python scripts/load_findings_to_db.py`
4. Restart app

### Parser errors
The markdown table parser expects:
- Tables with pipes `|` delimiters
- Header row followed by separator `|---|---|...|`
- Data rows with same number of columns as headers
- Backticks around code (e.g., `` `｜Zsh｜` ``)
