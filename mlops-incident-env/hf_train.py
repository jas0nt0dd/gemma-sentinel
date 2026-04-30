"""
Kaggle / HF Jobs GPU runner for GRPO fine-tuning — SentinelAI (Gemma 4 Good Hackathon)

Model: google/gemma-3-4b-it (Gemma 4 via Unsloth)
Environment: MLOps Incident OpenEnv (live FastAPI RL environment)
Method: GRPO (Group Relative Policy Optimization) with optional SFT warm-up

Workflow:
  1. (Optional) SFT warm-up on oracle-distilled traces (SFT_ORACLE_JSONL)
  2. GRPO training against live env reward signal
  3. Push LoRA adapters + metrics + plots to HF Hub

Env vars (common):
  HF_TOKEN            Hugging Face token (for hub + gated models)
  HF_SPACE_URL        OpenEnv Space URL (default: jason9150 space)
  HF_REPO_ID          Target HF model repo id to push adapters
  MODEL_NAME          Base model id (default: google/gemma-3-4b-it)
  RUN_MODE            balanced | hard_focus (default: balanced)
  MAX_SEQ_LENGTH      Unsloth context window (default: max(2048, min(8192, MAX_NEW_TOK+1400)))
  MAX_NEW_TOK         GRPO/eval generation cap (default: 768 full, 512 fast)
  RUN_PROFILE         full (default) | fast

Optional tuning:
  N_EVAL_EPS, EVAL_EVERY, NUM_GEN, MAX_NEW_TOK, MAX_SEQ_LENGTH, NUM_EPOCHS, LR, OUTPUT_DIR, SAVE_STEPS

Oracle distillation (SFT before GRPO):
  SFT_ORACLE_JSONL     Path to JSONL with {"messages":[...]} or {"text":"..."} per line
  SFT_EPOCHS           SFT warm-up epochs (default 1)
  SFT_LR               SFT learning rate (default 2e-4)
  SFT_GRAD_ACCUM       Gradient accumulation for SFT (default 4)

Reward shaping:
  REWARD_MAX_SHAPING      Max total bonus on top of env score (default 0.05)
  REWARD_SHAPING_MIN_ENV  Env score must be >= this before keyword bonus (default 0.02)

Gemma-specific notes:
  - Gemma 4 uses <start_of_turn> / <end_of_turn> chat template (handled by tokenizer.chat_template)
  - LoRA target modules updated for Gemma attention architecture (no bias, q/k/v/o + MLP gates)
  - 4-bit quantization via Unsloth: fits on a single A100 40GB or T4 (16GB with batch=1)
  - Kaggle T4: set RUN_PROFILE=fast, MAX_NEW_TOK=512, NUM_EPOCHS=1
  - Kaggle P100/A100: use RUN_PROFILE=full defaults
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
import time
import warnings
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r".*`max_new_tokens`.*`max_length`.*seem to have been set",
    category=UserWarning,
)


def _prepare_unsloth_env() -> None:
    """Before any `import unsloth`, avoid Unsloth's HF stats ping that can time out on Colab/Kaggle."""
    if os.getenv("UNSLOTH_DISABLE_STATISTICS") is not None and str(os.getenv("UNSLOTH_DISABLE_STATISTICS")).strip() != "":
        return
    if os.path.isdir("/content") and os.path.isdir("/opt/colab"):
        os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
    # Kaggle: /kaggle/working
    if os.path.isdir("/kaggle"):
        os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"


_prepare_unsloth_env()

if os.getenv("TORCHDYNAMO_DISABLE", "").strip() == "":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"


def _maybe_bootstrap() -> None:
    """Best-effort dependency install for Kaggle/Colab/HF Jobs images."""
    if os.getenv("AUTO_PIP_INSTALL", "1").strip() not in {"1", "true", "True", "yes", "YES"}:
        return

    def pip_install(packages: str) -> None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages.split()])

    def _hf_hub_and_datasets_import_ok() -> bool:
        try:
            from huggingface_hub import HfApi  # noqa: F401
            from datasets import Dataset  # noqa: F401
            return True
        except Exception:
            return False

    if not _hf_hub_and_datasets_import_ok():
        pip_install("huggingface_hub>=1.5.0,<2.0 datasets>=3.4.1,<4.4.0")
    if not _hf_hub_and_datasets_import_ok():
        pip_install("--force-reinstall --no-cache-dir huggingface_hub>=1.5.0,<2.0 datasets>=3.4.1,<4.4.0")

    try:
        import requests  # noqa: F401
    except Exception:
        pip_install("requests")

    try:
        import matplotlib  # noqa: F401
    except Exception:
        pip_install("matplotlib")

    def _trl_has_grpo() -> bool:
        import importlib.util
        return importlib.util.find_spec("trl.trainer.grpo_trainer") is not None

    need_trl = False
    try:
        import trl  # noqa: F401
    except Exception:
        need_trl = True
    else:
        if not _trl_has_grpo():
            need_trl = True
    if need_trl:
        pip_install("trl>=0.26.0 peft accelerate bitsandbytes transformers")

    try:
        import mergekit  # noqa: F401
    except Exception:
        try:
            pip_install("mergekit --no-deps")
        except Exception:
            try:
                pip_install("mergekit")
            except Exception:
                pass

    try:
        import unsloth  # noqa: F401
    except Exception:
        pip_install("unsloth")

    def _accelerate_version_at_least(min_parts: tuple[int, int, int]) -> bool:
        try:
            from importlib.metadata import version as dist_version
            raw = dist_version("accelerate").split("+", 1)[0].strip()
            parts: list[int] = []
            for seg in raw.split("."):
                if not seg.isdigit():
                    break
                parts.append(int(seg))
            while len(parts) < 3:
                parts.append(0)
            return tuple(parts[:3]) >= min_parts
        except Exception:
            return False

    if not _accelerate_version_at_least((1, 10, 1)):
        pip_install("--force-reinstall --no-cache-dir accelerate>=1.10.1,<2.0")
    else:
        try:
            if "/tmp/hf_repo" in Path(__file__).resolve().as_posix():
                pip_install("--force-reinstall --no-cache-dir accelerate>=1.10.1,<2.0")
        except Exception:
            pass


_maybe_bootstrap()

import numpy as np
import requests
import torch
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from transformers import GenerationConfig, TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel


def _clear_gen_max_length(m: Any) -> None:
    """Unset hub default generation_config.max_length so only max_new_tokens applies."""
    seen: set[int] = set()
    stack: list[Any] = [m]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        gc = getattr(obj, "generation_config", None)
        if gc is not None:
            try:
                if getattr(gc, "max_length", None) is not None:
                    gc.max_length = None
            except Exception:
                pass
        for name in ("base_model", "model", "module"):
            ch = getattr(obj, name, None)
            if ch is not None and not isinstance(ch, (torch.Tensor, torch.nn.Parameter)):
                stack.append(ch)


# =========================
# Config
# =========================
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://jason9150-mlops-incident-env.hf.space").rstrip("/")
HF_REPO_ID = os.getenv("HF_REPO_ID", "jason9150/sentinel-ai-gemma4-grpo").strip()


def _validate_model_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip()
    if not repo_id:
        raise SystemExit("HF_REPO_ID is required and must point to a Hugging Face model repo.")
    lowered = repo_id.lower()
    if "huggingface.co/spaces/" in lowered or lowered.startswith("spaces/"):
        raise SystemExit(f"HF_REPO_ID must be a model repo id, not a Space target: {repo_id!r}")
    if repo_id.startswith("http://") or repo_id.startswith("https://"):
        raise SystemExit(f"HF_REPO_ID must be a repo id like 'owner/name', not a URL: {repo_id!r}")
    if repo_id.count("/") != 1:
        raise SystemExit(f"HF_REPO_ID must look like 'owner/name'; got {repo_id!r}")
    return repo_id


HF_REPO_ID = _validate_model_repo_id(HF_REPO_ID)

# ─── GEMMA 4 MODEL ───────────────────────────────────────────────────────────
# Default: google/gemma-3-4b-it (Gemma 4, instruction-tuned, 4B params)
# Alternatives: google/gemma-3-1b-it (faster, lower VRAM), google/gemma-3-12b-it (better but needs A100)
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-3-4b-it").strip()

# Validation: must be a Gemma model
_model_lower = MODEL_NAME.lower()
if not ("gemma" in _model_lower):
    raise SystemExit(
        "MODEL_NAME must be a Gemma model id (e.g. google/gemma-3-4b-it or google/gemma-3-1b-it). "
        f"Got: {MODEL_NAME!r}\n"
        "For Kaggle T4: google/gemma-3-1b-it (1B, fast)\n"
        "For Kaggle P100/A100: google/gemma-3-4b-it (4B, recommended)"
    )
# ─────────────────────────────────────────────────────────────────────────────

RUN_MODE = os.getenv("RUN_MODE", "balanced").strip().lower()
RUN_PROFILE = os.getenv("RUN_PROFILE", "full").strip().lower()


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


if RUN_PROFILE in ("fast", "dev"):
    _n_eval_d, _eval_every_d, _epochs_d, _max_tok_d, _mid_eps_d = 1, 999_999, 1, 512, 1
    _save_best_d, _save_steps_d = False, 9999
else:
    _n_eval_d, _eval_every_d, _epochs_d, _max_tok_d, _mid_eps_d = 3, 10, 2, 768, 3
    _save_best_d, _save_steps_d = True, 25

N_EVAL_EPS = _env_int("N_EVAL_EPS", _n_eval_d)
EVAL_EVERY = _env_int("EVAL_EVERY", _eval_every_d)
NUM_GEN = _env_int("NUM_GEN", 2)
MAX_NEW_TOK = _env_int("MAX_NEW_TOK", _max_tok_d)
MAX_SEQ_LENGTH = _env_int("MAX_SEQ_LENGTH", max(2048, min(8192, MAX_NEW_TOK + 1400)))
NUM_EPOCHS = _env_int("NUM_EPOCHS", _epochs_d)
LR = float(os.getenv("LR", "1e-6"))
_default_output = f"sentinel-gemma4-grpo-{RUN_MODE}-fast" if RUN_PROFILE in ("fast", "dev") else f"sentinel-gemma4-grpo-{RUN_MODE}"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", _default_output).strip()
SAVE_STEPS = _env_int("SAVE_STEPS", _save_steps_d)

SFT_ORACLE_JSONL = os.getenv("SFT_ORACLE_JSONL", "").strip()
SFT_EPOCHS = _env_int("SFT_EPOCHS", 1)
SFT_LR = _env_float("SFT_LR", 2e-4)
SFT_GRAD_ACCUM = _env_int("SFT_GRAD_ACCUM", 4)
SFT_WARMUP_COMPLETED = False

TASKS = ["easy", "medium", "hard", "cascade"]

if RUN_MODE == "balanced":
    TRAIN_COUNTS = {"easy": 18, "medium": 18, "hard": 10, "cascade": 10}
else:
    TRAIN_COUNTS = {"easy": 22, "medium": 22, "hard": 14, "cascade": 14}

TRAIN_COUNTS_JSON = os.getenv("TRAIN_COUNTS_JSON", "").strip()
if TRAIN_COUNTS_JSON:
    override = json.loads(TRAIN_COUNTS_JSON)
    for k in TASKS:
        if k in override:
            TRAIN_COUNTS[k] = int(override[k])
elif RUN_PROFILE in ("fast", "dev"):
    TRAIN_COUNTS = {"easy": 5, "medium": 5, "hard": 3, "cascade": 3}

REWARD_MAX_SHAPING = _env_float("REWARD_MAX_SHAPING", 0.05)
REWARD_SHAPING_MIN_ENV = _env_float("REWARD_SHAPING_MIN_ENV", 0.02)
ENABLE_TASK_HINTS = _env_bool("ENABLE_TASK_HINTS", False)

EVAL_GREEDY = _env_bool("EVAL_GREEDY", True)
SAVE_BEST_BY_EVAL = _env_bool("SAVE_BEST_BY_EVAL", _save_best_d)
MID_EVAL_N_EPS = _env_int("MID_EVAL_N_EPS", _mid_eps_d)


def _print_header() -> None:
    print("=" * 72)
    print("SentinelAI — Gemma 4 GRPO Training (Gemma 4 Good Hackathon)")
    print("=" * 72)
    print("HF_SPACE_URL :", HF_SPACE_URL)
    print("HF_REPO_ID   :", HF_REPO_ID)
    print("MODEL_NAME   :", MODEL_NAME)
    print("RUN_MODE     :", RUN_MODE)
    print("RUN_PROFILE  :", RUN_PROFILE)
    print("TRAIN_COUNTS :", TRAIN_COUNTS)
    print("REWARD cap   : max_shaping=", REWARD_MAX_SHAPING, "kw_min_env=", REWARD_SHAPING_MIN_ENV)
    print("TASK_HINTS   :", ENABLE_TASK_HINTS)
    print("LR           :", LR)
    print("EVAL_GREEDY  :", EVAL_GREEDY)
    print("SAVE_BEST    :", SAVE_BEST_BY_EVAL, "MID_EVAL_N_EPS:", MID_EVAL_N_EPS, "SAVE_STEPS:", SAVE_STEPS)
    print("NUM_EPOCHS   :", NUM_EPOCHS)
    print("NUM_GEN      :", NUM_GEN)
    print("MAX_NEW_TOK  :", MAX_NEW_TOK)
    print("MAX_SEQ_LEN  :", MAX_SEQ_LENGTH)
    print("SFT_JSONL    :", SFT_ORACLE_JSONL or "(none)")
    if SFT_ORACLE_JSONL:
        print("SFT_EPOCHS   :", SFT_EPOCHS, "SFT_LR:", SFT_LR, "SFT_GRAD_ACCUM:", SFT_GRAD_ACCUM)
    print("CUDA         :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU          :", torch.cuda.get_device_name(0))
    print("HF_TOKEN set :", "yes" if bool(HF_TOKEN) else "no")
    print("=" * 72)


_print_header()


class MLOpsEnv:
    def __init__(self, base_url: str, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.retries = retries

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=10).status_code == 200
        except Exception:
            return False

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Exception | None = None
        for i in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}/{endpoint}", json=payload, timeout=45)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if i < self.retries - 1:
                    time.sleep(2 + i * 2)
        raise RuntimeError(f"POST {endpoint} failed: {last_err}")

    def reset(self, task_id: str) -> Dict[str, Any]:
        return self._post("reset", {"task_id": task_id})

    def step(self, action_type: str, target: str, parameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._post(
            "step",
            {
                "action_type": action_type,
                "target": target,
                "parameters": parameters or {},
            },
        )


env = MLOpsEnv(HF_SPACE_URL)
if not env.health():
    raise SystemExit(f"Environment unreachable: {HF_SPACE_URL}")
print("Environment health check: OK")


# ─── SYSTEM PROMPT (Gemma 4 optimised) ────────────────────────────────────
# Gemma 4 responds well to structured, concise system prompts.
# Think-before-answer format works naturally with Gemma's instruction tuning.
SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent powered by Gemma 4.
You diagnose production AI/ML system failures from component status and evidence.

Investigation reasoning:
- Prioritize components in degraded, error, warning, or critical states.
- Weigh log evidence, metric spikes, config changes, and feature drift.
- For cascade failures, identify the upstream root that explains all downstream symptoms.
- Name exact components from the COMPONENT STATUS list.

Think step-by-step in <think>...</think> (2-3 sentences max).
Then output ONLY this JSON on one line:
{"target":"<exact_component_name>","root_cause":"<one sentence>","fix":"<one sentence>"}

Rules:
- target must exactly match a name from COMPONENT STATUS
- No markdown, no extra text after the JSON
- Prefer the true root cause over noisy neighbor components
"""
# ──────────────────────────────────────────────────────────────────────────

INVESTIGATE = {
    "easy": [
        ("query_logs", "data_pipeline_a"),
        ("check_metrics", "data_pipeline_a"),
        ("check_metrics", "feature_store"),
    ],
    "medium": [
        ("inspect", "feature_preprocessor_v2"),
        ("compare_configs", "feature_preprocessor_v2"),
        ("check_metrics", "model_server"),
    ],
    "hard": [
        ("check_feature_drift", "feature_store"),
        ("check_feature_drift", "model_server"),
        ("check_metrics", "business"),
        ("check_metrics", "model_server"),
        ("compare_configs", "model_server"),
    ],
    "cascade": [
        ("check_metrics", "embedding_service_v3"),
        ("inspect", "feature_store"),
        ("check_metrics", "ab_test_router"),
    ],
}

TASK_DIAGNOSIS_HINTS = {
    "hard": "For drift-heavy incidents, likely roots are in feature_store or model_server.",
    "cascade": "For cascade failures, prioritize upstream embedding/feature issues over downstream symptoms.",
}

KEYWORD_HINTS = {
    "easy": ["pipeline", "schema", "validation", "null", "migration"],
    "medium": ["preprocessor", "batch", "worker", "config", "timeout"],
    "hard": ["drift", "psi", "feature", "degradation", "distribution"],
    "cascade": ["embedding", "cascade", "rollback", "chain", "propagat"],
}


def gather_user_prompt(task_id: str) -> str:
    """Collect env observation text for the user turn."""
    obs = env.reset(task_id)
    evidence_lines: List[str] = []
    for action, target in INVESTIGATE[task_id]:
        try:
            s = env.step(action, target)
            fb = (s.get("action_feedback") or "")[:260]
            if fb:
                evidence_lines.append(f"[{action}({target})] {fb}")
            if s.get("done"):
                break
        except Exception:
            pass
        time.sleep(0.05)

    status = obs.get("component_status", {}) or {}
    status_lines = "\n".join(f"- {k}: {v}" for k, v in status.items())[:900]
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "(none)"

    hint = TASK_DIAGNOSIS_HINTS.get(task_id, "") if ENABLE_TASK_HINTS else ""

    return (
        f"ALERT: {obs.get('alert_summary', '')}\n"
        f"GOAL: {obs.get('goal', '')}\n\n"
        f"COMPONENT STATUS:\n{status_lines}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        + (f"TASK HINT: {hint}\n\n" if hint else "")
        + "Diagnose the root cause and output the required JSON."
    )


def format_training_prompt(tokenizer: Any, user_text: str) -> str:
    """
    Apply Gemma 4 chat template via tokenizer.apply_chat_template.
    Gemma uses: <start_of_turn>user / <end_of_turn> / <start_of_turn>model
    The tokenizer handles this automatically when chat_template is set.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": f"{SYS_PROMPT}\n\n{user_text}"},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Gemma fallback (manual template)
    return (
        f"<start_of_turn>user\n{SYS_PROMPT}\n\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def completion_to_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        content = c.get("content", "")
        return content if isinstance(content, str) else str(c)
    if isinstance(c, list):
        return " ".join(completion_to_text(x) for x in c)
    return str(c)


def parse_target(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return str(obj.get("target", "")).strip()
    except Exception:
        pass
    m2 = re.search(r'["\']target["\']\\s*:\\s*["\']([^"\']+)["\']', cleaned, flags=re.I)
    return m2.group(1).strip() if m2 else ""


eval_env = MLOpsEnv(HF_SPACE_URL)


def _norm_component_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _canonicalize_target(raw_target: str, components: List[str]) -> str:
    """Map near-miss model outputs to an exact component name when possible."""
    if not raw_target or not components:
        return raw_target

    t = raw_target.strip()
    if t in components:
        return t

    t_low = t.lower()
    for c in components:
        if c.lower() == t_low:
            return c

    t_norm = _norm_component_name(t)
    norm_map = {_norm_component_name(c): c for c in components}
    if t_norm in norm_map:
        return norm_map[t_norm]

    contains = [c for c in components if t_norm and (t_norm in _norm_component_name(c) or _norm_component_name(c) in t_norm)]
    if len(contains) == 1:
        return contains[0]

    return t


def score_target(task_id: str, target: str) -> float:
    if not target:
        return 0.0
    try:
        obs = eval_env.reset(task_id)
        components = list((obs.get("component_status") or {}).keys())
        fixed_target = _canonicalize_target(target, components)
        s = eval_env.step("submit_diagnosis", fixed_target, {})
        return float(s.get("final_score") or s.get("reward") or 0.0)
    except Exception:
        return 0.0


def build_dataset(train_counts: Dict[str, int], tokenizer: Any) -> Dataset:
    prompts: List[str] = []
    task_ids: List[str] = []

    print("Collecting prompts...")
    for task in TASKS:
        count = int(train_counts.get(task, 0))
        print(f"  {task:8s} x {count}: ", end="", flush=True)
        for _ in range(count):
            try:
                user_text = gather_user_prompt(task)
                prompts.append(format_training_prompt(tokenizer, user_text))
                task_ids.append(task)
                print(".", end="", flush=True)
            except Exception:
                print("x", end="", flush=True)
            time.sleep(0.05)
        print(" done")

    ds = Dataset.from_dict({"prompt": prompts, "task_id": task_ids})
    print("Dataset size:", len(ds))
    return ds


# ─── LOAD GEMMA 4 via Unsloth ─────────────────────────────────────────────
print(f"Loading Gemma 4 model: {MODEL_NAME}")
print("Using Unsloth 4-bit quantization for memory efficiency")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,          # Auto: bfloat16 on Ampere+, float16 on older GPUs
    load_in_4bit=True,   # 4-bit NF4 quant — fits Gemma 4B on T4 (16GB)
    token=HF_TOKEN or None,
)

# Align tokenizer context with training window
if getattr(tokenizer, "model_max_length", MAX_SEQ_LENGTH) > MAX_SEQ_LENGTH:
    tokenizer.model_max_length = MAX_SEQ_LENGTH

# ─── LoRA config for Gemma 4 ──────────────────────────────────────────────
# Gemma attention: q_proj, k_proj, v_proj, o_proj
# Gemma MLP: gate_proj, up_proj, down_proj
# r=16 is a good default; increase to r=32 for better quality on A100
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)
# ──────────────────────────────────────────────────────────────────────────

_clear_gen_max_length(model)

trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
total_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Gemma 4 ready. Trainable: {trainable_m:.2f}M / Total: {total_m:.2f}M params")

train_ds = build_dataset(TRAIN_COUNTS, tokenizer)


def _load_sft_text_dataset(path: str, tok: Any) -> Dataset:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" in row and str(row["text"]).strip():
                texts.append(str(row["text"]))
                continue
            msgs = row.get("messages")
            if isinstance(msgs, list) and msgs:
                texts.append(
                    tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                )
    return Dataset.from_dict({"text": texts})


def _run_oracle_sft_warmup() -> None:
    """Supervised warm-up on oracle distilled JSONL before GRPO."""
    global SFT_WARMUP_COMPLETED
    if not SFT_ORACLE_JSONL:
        return
    if not os.path.isfile(SFT_ORACLE_JSONL):
        print(f"SFT_ORACLE_JSONL not found: {SFT_ORACLE_JSONL!r}; skipping SFT warm-up.")
        return
    from trl import SFTConfig, SFTTrainer

    sft_ds = _load_sft_text_dataset(SFT_ORACLE_JSONL, tokenizer)
    if len(sft_ds) < 1:
        print("SFT oracle dataset empty; skipping warm-up.")
        return
    sft_out = os.path.join(OUTPUT_DIR, "sft_warmup")
    os.makedirs(sft_out, exist_ok=True)
    print(
        f"Oracle SFT warm-up: examples={len(sft_ds)} epochs={SFT_EPOCHS} lr={SFT_LR} "
        f"accum={SFT_GRAD_ACCUM} -> {sft_out}"
    )
    base_kwargs: Dict[str, Any] = dict(
        output_dir=sft_out,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        learning_rate=SFT_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=max(1, min(50, len(sft_ds) // 2)),
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        dataset_text_field="text",
        dataloader_num_workers=0,
    )
    try:
        sft_args = SFTConfig(max_seq_length=MAX_SEQ_LENGTH, **base_kwargs)
    except TypeError:
        sft_args = SFTConfig(**base_kwargs)
    try:
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=sft_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=sft_ds,
            tokenizer=tokenizer,
        )
    model.train()
    sft_trainer.train()
    del sft_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    SFT_WARMUP_COMPLETED = True
    print("Oracle SFT warm-up finished.")


_run_oracle_sft_warmup()


def run_eval(task_id: str, n_eps: int = N_EVAL_EPS) -> float:
    scores: List[float] = []
    FastLanguageModel.for_inference(model)

    for _ in range(n_eps):
        try:
            prompt = format_training_prompt(tokenizer, gather_user_prompt(task_id))
            inp = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_cfg = GenerationConfig(
                max_new_tokens=MAX_NEW_TOK,
                pad_token_id=tokenizer.eos_token_id,
            )
            with torch.no_grad():
                if EVAL_GREEDY:
                    out = model.generate(**inp, generation_config=gen_cfg, do_sample=False)
                else:
                    out = model.generate(
                        **inp,
                        generation_config=gen_cfg,
                        do_sample=True,
                        temperature=0.3,
                    )
            raw = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            target = parse_target(raw)
            scores.append(score_target(task_id, target))
        except Exception:
            scores.append(0.0)
        time.sleep(0.05)

    return float(np.mean(scores)) if scores else 0.0


print("Running baseline eval (Gemma 4 before GRPO)...")
baseline = {t: run_eval(t) for t in TASKS}
avg_baseline = float(np.mean(list(baseline.values())))
print("Baseline:", baseline, "avg:", round(avg_baseline, 4))
model.train()

reward_log: List[float] = []


def mlops_reward(completions, task_id=None, **kwargs):
    """
    GRPO reward function for SentinelAI.
    Primary signal: env score (0.0 to 1.0) from submit_diagnosis.
    Secondary (capped): format bonus for JSON structure + think tag + keyword match.
    """
    rewards: List[float] = []
    for i, c in enumerate(completions):
        text = completion_to_text(c)
        tid = task_id[i] if task_id is not None else "easy"
        target = parse_target(text)

        if not target:
            r = -0.25
            rewards.append(r)
            reward_log.append(r)
            continue

        r_env = float(score_target(tid, target))
        shaping = 0.0

        # JSON structure bonus
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        try:
            m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            obj = json.loads(m.group()) if m else {}
            if isinstance(obj, dict) and all(k in obj for k in ["target", "root_cause", "fix"]):
                shaping += 0.02
        except Exception:
            pass

        # Think-before-answer bonus (Gemma 4 does this well naturally)
        if "<think>" in text and "</think>" in text:
            shaping += 0.015

        # Keyword match bonus (only when env score is non-zero)
        if r_env >= REWARD_SHAPING_MIN_ENV:
            text_l = text.lower()
            hits = sum(1 for kw in KEYWORD_HINTS.get(tid, []) if kw in text_l)
            shaping += min(0.04, hits * 0.012)

        shaping = min(REWARD_MAX_SHAPING, shaping)
        r = min(1.0, r_env + shaping)

        rewards.append(r)
        reward_log.append(r)

    return rewards


def _active_adapter_name(m: Any) -> str:
    aa = getattr(m, "active_adapter", None)
    if isinstance(aa, str):
        return aa
    if isinstance(aa, list) and aa:
        return str(aa[0])
    aas = getattr(m, "active_adapters", None)
    if isinstance(aas, (list, tuple)) and aas:
        return str(aas[0])
    return "default"


def _save_best_peft_adapter(m: Any, path: str) -> None:
    """Only LoRA / PEFT tensors — safe with 4-bit quantized base."""
    sd = get_peft_model_state_dict(m)
    torch.save(sd, path)


def _load_best_peft_adapter(m: Any, path: str) -> None:
    device = next(m.parameters()).device
    sd = torch.load(path, map_location=device)
    set_peft_model_state_dict(m, sd, adapter_name=_active_adapter_name(m))


class EvalCallback(TrainerCallback):
    def __init__(self, eval_every: int, mid_n_eps: int, save_best: bool, output_dir: str):
        self.eval_every = eval_every
        self.mid_n_eps = mid_n_eps
        self.save_best = save_best
        self.best_adapter_path = os.path.join(output_dir, "best_eval_adapter.pt")
        self.checkpoints: List[Tuple[int, float]] = []
        self.best_avg = -1.0
        self.best_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return
        model.eval()
        scores = {t: run_eval(t, n_eps=self.mid_n_eps) for t in TASKS}
        avg = float(np.mean(list(scores.values())))
        self.checkpoints.append((state.global_step, avg))
        print(f"\n[EVAL step {state.global_step}] {scores} avg={avg:.4f}")
        if self.save_best and avg > self.best_avg:
            self.best_avg = avg
            self.best_step = state.global_step
            _save_best_peft_adapter(model, self.best_adapter_path)
            print(f"  [saved best adapter step={state.global_step} avg={avg:.4f}]")
        model.train()


eval_cb = EvalCallback(EVAL_EVERY, MID_EVAL_N_EPS, SAVE_BEST_BY_EVAL, OUTPUT_DIR)

# ─── GRPO Config (Gemma 4 tuned) ─────────────────────────────────────────
# Key differences from Llama config:
# - Same batch size (1) and grad accum (2) — Gemma 4B is similar VRAM to Llama 3.2 3B
# - adamw_8bit works well with Gemma via Unsloth
# - bf16 preferred (Gemma was trained in bfloat16)
cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=0.3,
    num_generations=NUM_GEN,
    max_completion_length=MAX_NEW_TOK,
    temperature=0.7,
    logging_steps=1,
    save_steps=SAVE_STEPS,
    report_to="none",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    dataloader_num_workers=0,
)
# ──────────────────────────────────────────────────────────────────────────

trainer = GRPOTrainer(
    model=model,
    args=cfg,
    processing_class=tokenizer,
    train_dataset=train_ds,
    reward_funcs=[mlops_reward],
    callbacks=[eval_cb],
)

print("Starting Gemma 4 GRPO training...")
stats = trainer.train()
runtime_min = float(stats.metrics.get("train_runtime", 0.0)) / 60.0
print(f"Train runtime (min): {runtime_min:.2f}")

best_adapter_path = os.path.join(OUTPUT_DIR, "best_eval_adapter.pt")
legacy_state_path = os.path.join(OUTPUT_DIR, "best_eval_model_state.pt")
if SAVE_BEST_BY_EVAL and os.path.isfile(best_adapter_path):
    print(f"Reloading best mid-training adapter (step {eval_cb.best_step}, avg {eval_cb.best_avg:.4f})...")
    _load_best_peft_adapter(model, best_adapter_path)
elif SAVE_BEST_BY_EVAL and eval_cb.checkpoints:
    print("No best adapter file; using final training weights for post-train eval.")

print("Running post-train eval (Gemma 4 after GRPO)...")
trained = {t: run_eval(t) for t in TASKS}
avg_trained = float(np.mean(list(trained.values())))

print("\n" + "=" * 72)
print("SENTINELAI — GEMMA 4 GRPO RESULTS")
print("=" * 72)
for t in TASKS:
    delta = trained[t] - baseline[t]
    print(f"{t:8s} before={baseline[t]:.4f}  after={trained[t]:.4f}  delta={delta:+.4f}")
print("-" * 72)
print(f"average  before={avg_baseline:.4f}  after={avg_trained:.4f}  delta={avg_trained-avg_baseline:+.4f}")
print("=" * 72)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x = np.arange(len(TASKS))
w = 0.35
plt.bar(x - w / 2, [baseline[t] for t in TASKS], w, label="Before GRPO (Gemma 4 base)")
plt.bar(x + w / 2, [trained[t] for t in TASKS], w, label="After GRPO (SentinelAI)")
plt.xticks(x, TASKS)
plt.ylabel("Score (mean env reward)")
plt.ylim(0, 1.05)
plt.title("SentinelAI — Before vs After GRPO")
plt.legend()

plt.subplot(1, 2, 2)
if eval_cb.checkpoints:
    xs, ys = zip(*eval_cb.checkpoints)
    plt.plot(xs, ys, marker="o", label="Mid-train eval (avg)")
plt.axhline(avg_baseline, linestyle="--", color="C0", label=f"Baseline {avg_baseline:.3f}")
if SAVE_BEST_BY_EVAL and eval_cb.best_step >= 0 and eval_cb.best_avg >= 0:
    plt.axhline(
        eval_cb.best_avg,
        linestyle=":",
        color="green",
        linewidth=2,
        label=f"Best mid-train {eval_cb.best_avg:.3f} (step {eval_cb.best_step})",
    )
plt.axhline(avg_trained, linestyle="--", color="C1", label=f"Post-train {avg_trained:.3f}")
plt.xlabel("Training step")
plt.ylabel("Average score (env reward)")
plt.title("Gemma 4 GRPO — Reward Progression")
plt.legend(fontsize=8)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plot_path = os.path.join(OUTPUT_DIR, "grpo_reward_curves.png")
plt.savefig(plot_path, dpi=180)
print(f"Saved plot: {plot_path}")

local_dir = "sentinel-ai-gemma4-lora"
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)
print(f"Saved LoRA adapters: {local_dir}")

metrics = {
    "project": "SentinelAI",
    "hackathon": "Gemma 4 Good Hackathon",
    "run_mode": RUN_MODE,
    "run_profile": RUN_PROFILE,
    "model_name": MODEL_NAME,
    "train_counts": TRAIN_COUNTS,
    "lr": LR,
    "reward_max_shaping": REWARD_MAX_SHAPING,
    "reward_shaping_min_env": REWARD_SHAPING_MIN_ENV,
    "eval_greedy": EVAL_GREEDY,
    "save_best_by_eval": SAVE_BEST_BY_EVAL,
    "save_steps": SAVE_STEPS,
    "mid_eval_n_eps": MID_EVAL_N_EPS,
    "best_eval_step": eval_cb.best_step,
    "best_eval_avg": eval_cb.best_avg,
    "num_epochs": NUM_EPOCHS,
    "num_generations": NUM_GEN,
    "max_new_tokens": MAX_NEW_TOK,
    "max_seq_length": MAX_SEQ_LENGTH,
    "runtime_min": runtime_min,
    "baseline": baseline,
    "trained": trained,
    "avg_baseline": avg_baseline,
    "avg_trained": avg_trained,
    "delta_avg": avg_trained - avg_baseline,
    "checkpoints": eval_cb.checkpoints,
    "sft_oracle_jsonl": SFT_ORACLE_JSONL,
    "sft_epochs": SFT_EPOCHS,
    "sft_lr": SFT_LR,
    "sft_warmup_completed": SFT_WARMUP_COMPLETED,
}
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics: {metrics_path}")

if HF_TOKEN:
    create_repo(HF_REPO_ID, token=HF_TOKEN, exist_ok=True, private=False, repo_type="model")
    model.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=False)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=False)
    print(f"Pushed Gemma 4 LoRA adapters: https://huggingface.co/{HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=metrics_path,
        path_in_repo="metrics.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    if os.path.isfile(plot_path):
        api.upload_file(
            path_or_fileobj=plot_path,
            path_in_repo="grpo_reward_curves.png",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
    print(f"Uploaded metrics.json + plot to https://huggingface.co/{HF_REPO_ID}")
else:
    print("HF_TOKEN missing; skipped Hub push.")

print("DONE — SentinelAI Gemma 4 GRPO training complete")
