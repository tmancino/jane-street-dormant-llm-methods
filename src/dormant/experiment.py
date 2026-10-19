"""Experiment — Isolated pipeline run state management.

Each experiment gets its own directory under data/experiments/{id}/ containing:
  - config.json      — experiment metadata, parameters, creation time
  - state.json       — pipeline stage status tracking
  - probes.json      — the probe set for this experiment
  - responses/       — per-model API responses (chat + activations)
  - analysis/        — derived analysis results
  - logs/            — run logs

No cross-pollination between experiments. Each is fully self-contained.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.parent / "data" / "experiments"

PIPELINE_STAGES = [
    "probes_generated",      # Probe set created (from SPOCK or manual)
    "baseline_collected",    # Clean model baseline responses
    "m1_chat_submitted",     # Chat batch submitted to M1
    "m1_chat_collected",     # Chat results downloaded for M1
    "m1_act_submitted",      # Activation batch submitted to M1
    "m1_act_collected",      # Activation results downloaded for M1
    "m2_chat_submitted",
    "m2_chat_collected",
    "m2_act_submitted",
    "m2_act_collected",
    "m3_chat_submitted",
    "m3_chat_collected",
    "m3_act_submitted",
    "m3_act_collected",
    "cross_model_analyzed",  # Cross-model divergence computed
]

MODEL_IDS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
MODEL_SHORT = {"dormant-model-1": "M1", "dormant-model-2": "M2", "dormant-model-3": "M3"}


class Experiment:
    """Manages an isolated experiment run."""

    def __init__(self, experiment_id: str):
        self.id = experiment_id
        self.dir = EXPERIMENTS_DIR / experiment_id
        self.config_path = self.dir / "config.json"
        self.state_path = self.dir / "state.json"
        self.kirk_state_path = self.dir / "kirk_state.json"
        self.probes_path = self.dir / "probes.json"
        self.responses_dir = self.dir / "responses"
        self.analysis_dir = self.dir / "analysis"
        self.results_dir = self.dir / "results"
        self.logs_dir = self.dir / "logs"

    # ── Creation ──────────────────────────────────────────────────────

    @classmethod
    def create(cls, name: str, description: str = "",
               activation_layers: list = None,
               activation_modules: list = None) -> "Experiment":
        """Create a new experiment with a timestamped ID."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{ts}_{name}"
        exp = cls(experiment_id)

        # Create directory structure
        exp.dir.mkdir(parents=True, exist_ok=True)
        exp.responses_dir.mkdir(exist_ok=True)
        exp.analysis_dir.mkdir(exist_ok=True)
        exp.logs_dir.mkdir(exist_ok=True)
        exp.ensure_results_dirs()

        # Default activation modules: model.norm + gate at key layers
        if activation_layers is None:
            activation_layers = [20, 36, 37, 40, 57, 58]
        if activation_modules is None:
            activation_modules = ["model.norm"]
            for layer in activation_layers:
                activation_modules.append(f"model.layers.{layer}.mlp.gate")

        config = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "activation_layers": activation_layers,
            "activation_modules": activation_modules,
            "models": MODEL_IDS,
        }
        with open(exp.config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Initialize empty state
        state = {stage: {"status": "pending"} for stage in PIPELINE_STAGES}
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["batch_ids"] = {}
        with open(exp.state_path, "w") as f:
            json.dump(state, f, indent=2)

        return exp

    @classmethod
    def load(cls, experiment_id: str) -> "Experiment":
        """Load an existing experiment."""
        exp = cls(experiment_id)
        if not exp.config_path.exists():
            raise FileNotFoundError(
                f"Experiment {experiment_id} not found at {exp.dir}")
        return exp

    @classmethod
    def latest(cls) -> "Experiment":
        """Load the most recently created experiment."""
        if not EXPERIMENTS_DIR.exists():
            raise FileNotFoundError("No experiments directory")
        dirs = sorted(EXPERIMENTS_DIR.iterdir(), reverse=True)
        for d in dirs:
            if (d / "config.json").exists():
                return cls.load(d.name)
        raise FileNotFoundError("No experiments found")

    @classmethod
    def list_all(cls) -> list:
        """List all experiment IDs."""
        if not EXPERIMENTS_DIR.exists():
            return []
        return sorted(
            [d.name for d in EXPERIMENTS_DIR.iterdir()
             if (d / "config.json").exists()],
            reverse=True)

    # ── Config & State ────────────────────────────────────────────────

    @property
    def config(self) -> dict:
        with open(self.config_path) as f:
            return json.load(f)

    @property
    def state(self) -> dict:
        with open(self.state_path) as f:
            return json.load(f)

    def update_state(self, stage: str, status: str, **kwargs):
        """Update a pipeline stage status."""
        state = self.state
        if stage not in state:
            state[stage] = {}
        state[stage]["status"] = status
        state[stage]["updated_at"] = datetime.now(timezone.utc).isoformat()
        state[stage].update(kwargs)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ── Results Directories ──────────────────────────────────────────

    def ensure_results_dirs(self):
        """Create results subdirectories for diana_troy and tricorder output."""
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "diana_troy").mkdir(exist_ok=True)
        (self.results_dir / "tricorder").mkdir(exist_ok=True)

    @property
    def diana_troy_dir(self) -> Path:
        return self.results_dir / "diana_troy"

    @property
    def tricorder_dir(self) -> Path:
        return self.results_dir / "tricorder"

    # ── Kirk State ─────────────────────────────────────────────────

    def read_kirk_state(self) -> dict:
        """Read experiment-scoped kirk_state.json with file locking."""
        import fcntl
        if not self.kirk_state_path.exists():
            return {
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "models": {},
                "climbing": {},
                "log": [],
                "experiment_id": self.id,
            }
        try:
            with open(self.kirk_state_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError):
            return {
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "models": {},
                "climbing": {},
                "log": [],
                "experiment_id": self.id,
            }

    def write_kirk_state(self, state: dict):
        """Write experiment-scoped kirk_state.json atomically with locking."""
        import fcntl
        state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state["experiment_id"] = self.id
        tmp = self.kirk_state_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2, ensure_ascii=False)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        tmp.replace(self.kirk_state_path)

    def store_batch_id(self, label: str, batch_id: str):
        """Store a batch ID for later polling."""
        state = self.state
        state["batch_ids"][label] = batch_id
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def get_batch_id(self, label: str) -> str:
        return self.state.get("batch_ids", {}).get(label)

    # ── Probes ────────────────────────────────────────────────────────

    def save_probes(self, probes: list):
        """Save probe set. Each probe: {id, text, source, expert_id, layer}."""
        data = {
            "n_probes": len(probes),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "probes": probes,
        }
        with open(self.probes_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_probes(self) -> list:
        if not self.probes_path.exists():
            return []
        with open(self.probes_path) as f:
            return json.load(f).get("probes", [])

    # ── Responses ─────────────────────────────────────────────────────

    def response_path(self, model_id: str, kind: str) -> Path:
        """Path for a model's response file. kind: 'chat' or 'act'."""
        short = MODEL_SHORT[model_id]
        return self.responses_dir / f"{short}_{kind}.json"

    def save_responses(self, model_id: str, kind: str, data: dict):
        path = self.response_path(model_id, kind)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_responses(self, model_id: str, kind: str) -> dict:
        path = self.response_path(model_id, kind)
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def has_responses(self, model_id: str, kind: str) -> bool:
        return self.response_path(model_id, kind).exists()

    # ── Analysis ──────────────────────────────────────────────────────

    def analysis_path(self, name: str) -> Path:
        return self.analysis_dir / f"{name}.json"

    def save_analysis(self, name: str, data: dict):
        path = self.analysis_path(name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_analysis(self, name: str) -> dict:
        path = self.analysis_path(name)
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    # ── Logging ───────────────────────────────────────────────────────

    def log(self, message: str):
        """Append a timestamped log entry."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log_path = self.logs_dir / "pipeline.log"
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {message}\n")

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable experiment summary."""
        cfg = self.config
        st = self.state
        lines = [
            f"Experiment: {self.id}",
            f"  Name: {cfg['name']}",
            f"  Created: {cfg['created_at']}",
            f"  Layers: {cfg['activation_layers']}",
            f"  Pipeline:",
        ]
        for stage in PIPELINE_STAGES:
            info = st.get(stage, {})
            status = info.get("status", "unknown")
            lines.append(f"    {stage}: {status}")
        probes = self.load_probes()
        lines.append(f"  Probes: {len(probes)}")
        for model_id in MODEL_IDS:
            short = MODEL_SHORT[model_id]
            chat = "YES" if self.has_responses(model_id, "chat") else "no"
            act = "YES" if self.has_responses(model_id, "act") else "no"
            lines.append(f"  {short}: chat={chat} act={act}")
        return "\n".join(lines)
