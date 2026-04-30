"""
SentinelAI — Gradio Demo App
Gemma 4 Good Hackathon | Global Resilience Impact Track

This is the live demo for HuggingFace Spaces.

How it works:
  1. User picks a task scenario (easy / medium / hard / cascade)
  2. SentinelAI (Gemma 4 via HF Inference API) reads the live MLOps env
  3. Agent investigates: queries logs, checks metrics, compares configs
  4. Gemma 4 reasons with <think> tags and outputs a structured JSON diagnosis
  5. Diagnosis is submitted to the env → real score returned

Env vars (set as HF Space secrets):
  HF_TOKEN        Your HuggingFace token (for Inference API)
  SPACE_URL       MLOps Incident env Space URL (default: jason9150 space)
  MODEL_ID        Gemma model to use (default: google/gemma-3-4b-it)
"""

import json
import os
import re
import time
from typing import Generator

import gradio as gr
import requests
from huggingface_hub import InferenceClient

# ─── Config ────────────────────────────────────────────────────────────
HF_TOKEN  = os.getenv("HF_TOKEN", "")
SPACE_URL = os.getenv("SPACE_URL", "https://jason9150-mlops-incident-env.hf.space").rstrip("/")
MODEL_ID  = os.getenv("MODEL_ID",  "google/gemma-3-4b-it")

# ─── Task metadata ──────────────────────────────────────────────────────────
TASK_META = {
    "easy": {
        "label": "🟢 Easy — Data Quality Alert",
        "desc": "A data pipeline is causing accuracy drops. Schema migration went wrong.",
        "investigate": [
            ("query_logs",    "data_pipeline_a"),
            ("check_metrics", "data_pipeline_a"),
            ("check_metrics", "feature_store"),
        ],
    },
    "medium": {
        "label": "🟡 Medium — Latency Spike",
        "desc": "Model serving latency spiked after a config change. Bottleneck unknown.",
        "investigate": [
            ("inspect",         "feature_preprocessor_v2"),
            ("compare_configs", "feature_preprocessor_v2"),
            ("check_metrics",   "model_server"),
        ],
    },
    "hard": {
        "label": "🔴 Hard — Silent Model Drift",
        "desc": "No alerts fired, but revenue dropped 12%. Feature distribution has shifted.",
        "investigate": [
            ("check_feature_drift", "feature_store"),
            ("check_feature_drift", "model_server"),
            ("check_metrics",       "business"),
            ("check_metrics",       "model_server"),
            ("compare_configs",     "model_server"),
        ],
    },
    "cascade": {
        "label": "☠️ Cascade — Multi-System Failure",
        "desc": "3 simultaneous failures triggered by one bad deployment. Full cascade investigation.",
        "investigate": [
            ("check_metrics", "embedding_service_v3"),
            ("inspect",       "feature_store"),
            ("check_metrics", "ab_test_router"),
        ],
    },
}

SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent powered by Gemma 4.
You diagnose production AI/ML system failures from component status and evidence.

Reasoning steps:
- Identify components in degraded, error, warning, or critical states.
- Weigh log evidence, metric spikes, config changes, and feature drift signals.
- For cascade failures, find the upstream root that explains all downstream symptoms.
- target must exactly match a component name from COMPONENT STATUS.

Think step-by-step in <think>...</think> (2-3 sentences).
Then output ONLY this single-line JSON:
{"target":"<exact_component_name>","root_cause":"<one sentence>","fix":"<one sentence>"}

No markdown. No extra text after the JSON."""


# ─── Environment helpers ──────────────────────────────────────────────────────────
def _post(endpoint: str, payload: dict, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.post(f"{SPACE_URL}/{endpoint}", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 + i)
    return {}


def env_reset(task_id: str) -> dict:
    return _post("reset", {"task_id": task_id})


def env_step(action: str, target: str) -> dict:
    return _post("step", {"action_type": action, "target": target, "parameters": {}})


def env_health() -> bool:
    try:
        return requests.get(f"{SPACE_URL}/health", timeout=8).status_code == 200
    except Exception:
        return False


def parse_json_from_text(text: str) -> dict | None:
    """Extract the JSON diagnosis from model output (strips <think> tags first)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


def extract_think(text: str) -> str:
    """Pull out the <think> block for display."""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


# ─── Main agent function ──────────────────────────────────────────────────────────
def run_sentinel(task_id: str) -> Generator[tuple, None, None]:
    """
    Full agent pipeline — yields (log, thinking, diagnosis, score, status) progressively.
    Gradio shows updates in real-time as each step completes.
    """
    log_lines = []
    thinking  = ""
    diagnosis = {}
    score     = 0.0
    status    = "⏳ Starting..."

    def emit():
        log_text  = "\n".join(log_lines)
        diag_text = json.dumps(diagnosis, indent=2) if diagnosis else ""
        score_md  = _score_badge(score) if diagnosis else ""
        return log_text, thinking, diag_text, score_md, status

    # ─ Step 1: Health check
    status = "🔍 Checking environment..."
    yield emit()
    if not env_health():
        status = "❌ Environment unreachable. Check SPACE_URL."
        log_lines.append(f"ERROR: Cannot reach {SPACE_URL}")
        yield emit()
        return
    log_lines.append(f"✅ Environment online: {SPACE_URL}")

    # ─ Step 2: Reset env + get initial observation
    status = "🔄 Resetting scenario..."
    yield emit()
    try:
        obs = env_reset(task_id)
    except Exception as e:
        status = f"❌ Reset failed: {e}"
        log_lines.append(f"ERROR: {e}")
        yield emit()
        return

    alert   = obs.get("alert_summary", "(no alert)")
    goal    = obs.get("goal",          "(no goal)")
    comp_st = obs.get("component_status", {}) or {}
    log_lines.append(f"\n🚨 ALERT: {alert}")
    log_lines.append(f"🎯 GOAL:  {goal}")
    log_lines.append(f"\n📊 COMPONENT STATUS:")
    for k, v in comp_st.items():
        icon = "🔴" if v in ("error", "critical") else "🟡" if v == "degraded" else "🟢"
        log_lines.append(f"  {icon} {k}: {v}")
    yield emit()

    # ─ Step 3: Investigation actions
    status = "🔍 Investigating..."
    evidence_lines = []
    investigate = TASK_META[task_id]["investigate"]

    for action, target in investigate:
        log_lines.append(f"\n➤ [{action}({target})]")
        yield emit()
        try:
            s   = env_step(action, target)
            fb  = (s.get("action_feedback") or "")[:400]
            if fb:
                log_lines.append(f"  ↳ {fb}")
                evidence_lines.append(f"[{action}({target})] {fb}")
            if s.get("done"):
                break
        except Exception as e:
            log_lines.append(f"  ⚠️ {e}")
        time.sleep(0.1)
        yield emit()

    # ─ Step 4: Build prompt for Gemma 4
    status = "🤖 Gemma 4 is reasoning..."
    yield emit()

    status_lines  = "\n".join(f"- {k}: {v}" for k, v in comp_st.items())[:900]
    evidence_text = "\n".join(evidence_lines) or "(none)"
    user_msg = (
        f"ALERT: {alert}\n"
        f"GOAL: {goal}\n\n"
        f"COMPONENT STATUS:\n{status_lines}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Diagnose the root cause and output the required JSON."
    )

    # ─ Step 5: Call Gemma 4 via HF Inference API
    raw_output = ""
    try:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not set. Add it as a Space secret.")

        client = InferenceClient(
            model=MODEL_ID,
            token=HF_TOKEN,
        )
        messages = [
            {"role": "user", "content": f"{SYS_PROMPT}\n\n{user_msg}"},
        ]
        # Stream tokens so users see Gemma thinking in real-time
        for chunk in client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            raw_output += delta
            # Update thinking display as tokens stream in
            thinking = extract_think(raw_output) or raw_output[:300]
            yield emit()

    except Exception as e:
        status = f"❌ Gemma 4 call failed: {e}"
        log_lines.append(f"\nERROR calling {MODEL_ID}: {e}")
        log_lines.append("Tip: Set HF_TOKEN as a Space secret with Inference API access.")
        yield emit()
        return

    log_lines.append(f"\n🤖 Raw Gemma 4 output:\n{raw_output[:600]}")
    thinking = extract_think(raw_output)
    yield emit()

    # ─ Step 6: Parse diagnosis
    status = "📊 Parsing diagnosis..."
    yield emit()

    parsed = parse_json_from_text(raw_output)
    if not parsed or "target" not in parsed:
        status = "⚠️ Could not parse JSON diagnosis from model output."
        log_lines.append("WARNING: No valid JSON found in model output.")
        yield emit()
        return

    diagnosis = parsed
    log_lines.append(f"\n✅ Parsed diagnosis: target={parsed['target']}")
    yield emit()

    # ─ Step 7: Submit to environment for scoring
    status = "🎯 Submitting diagnosis to environment..."
    yield emit()
    try:
        result = env_step("submit_diagnosis", parsed["target"])
        score  = float(result.get("final_score") or result.get("reward") or 0.0)
        fb     = result.get("action_feedback") or ""
        log_lines.append(f"\n🎯 SCORE: {score:.4f}")
        if fb:
            log_lines.append(f"  Feedback: {fb}")
    except Exception as e:
        log_lines.append(f"\nWARNING: Score submission failed: {e}")
        score = 0.0

    status = _status_from_score(score)
    yield emit()


def _score_badge(score: float) -> str:
    """Return a Markdown badge for the score."""
    if score >= 0.9:
        color, label = "#22c55e", "EXCELLENT"
    elif score >= 0.7:
        color, label = "#84cc16", "GOOD"
    elif score >= 0.4:
        color, label = "#f59e0b", "PARTIAL"
    elif score > 0.0:
        color, label = "#ef4444", "LOW"
    else:
        color, label = "#6b7280", "MISS"
    pct = int(score * 100)
    return (
        f'<div style="display:inline-block;background:{color};color:white;'
        f'padding:8px 20px;border-radius:8px;font-size:1.4rem;font-weight:700;">"
        f"🎯 {pct}% — {label}</div>"
    )


def _status_from_score(score: float) -> str:
    if score >= 0.9:
        return "✅ Perfect diagnosis! SentinelAI nailed it."
    elif score >= 0.7:
        return "🟢 Good diagnosis — correct root identified."
    elif score >= 0.4:
        return "🟡 Partial credit — partially correct."
    elif score > 0.0:
        return "🔴 Low score — wrong component targeted."
    else:
        return "❌ Miss — incorrect diagnosis."


# ─── Gradio UI ────────────────────────────────────────────────────────────
TASK_CHOICES = [(v["label"], k) for k, v in TASK_META.items()]

CSS = """
.sentinel-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2d4a 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 16px;
    border: 1px solid #334155;
}
.sentinel-header h1 {
    color: #38bdf8;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.sentinel-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.95rem;
}
.task-desc {
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 8px;
}
.log-box {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
}
"""

with gr.Blocks(
    title="SentinelAI — MLOps Incident Response Agent",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="sky",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as demo:

    # ─ Header
    gr.HTML("""
    <div class="sentinel-header">
        <h1>🛡️ SentinelAI</h1>
        <p>Autonomous MLOps Incident Response · Powered by Gemma 4 · Gemma 4 Good Hackathon</p>
    </div>
    """)

    with gr.Row():
        # ─ Left column: controls
        with gr.Column(scale=1):
            task_radio = gr.Radio(
                choices=TASK_CHOICES,
                value="easy",
                label="🎯 Select Incident Scenario",
                info="Each scenario has different complexity and failure patterns.",
            )
            task_desc_box = gr.HTML(
                value=f'<div class="task-desc">{TASK_META["easy"]["desc"]}</div>'
            )
            run_btn = gr.Button(
                "🚀 Run SentinelAI",
                variant="primary",
                size="lg",
            )

            gr.Markdown("""---
**How it works:**
1. Agent resets the live RL environment
2. Runs investigation actions (query logs, check metrics, etc.)
3. Gemma 4 reads all evidence and reasons
4. Outputs structured JSON diagnosis
5. Score returned from the live environment

**Model:** `google/gemma-3-4b-it` (Gemma 4)
**Env:** [MLOps Incident OpenEnv](https://huggingface.co/spaces/jason9150/mlops-incident-env)
""")

        # ─ Right column: outputs
        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="📡 Status",
                value="Select a scenario and click Run SentinelAI.",
                interactive=False,
                max_lines=1,
            )
            score_box = gr.HTML(label="🎯 Score")

            with gr.Tabs():
                with gr.TabItem("🔍 Investigation Log"):
                    log_box = gr.Textbox(
                        label="",
                        lines=18,
                        max_lines=30,
                        interactive=False,
                        elem_classes=["log-box"],
                        placeholder="Investigation log will appear here...",
                    )
                with gr.TabItem("🧠 Gemma 4 Reasoning"):
                    thinking_box = gr.Textbox(
                        label="",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        placeholder="Gemma 4's <think> reasoning will appear here as it streams...",
                    )
                with gr.TabItem("📊 Diagnosis JSON"):
                    diag_box = gr.Code(
                        label="",
                        language="json",
                        lines=8,
                        interactive=False,
                    )

    # ─ Examples section
    gr.Markdown("""---
### 📚 Scenario Guide

| Scenario | Difficulty | What to look for |
|---|---|---|
| 🟢 Easy — Data Quality | Beginner | Schema change in `data_pipeline_a` |
| 🟡 Medium — Latency Spike | Intermediate | Config change in `feature_preprocessor_v2` |
| 🔴 Hard — Silent Drift | Advanced | PSI spike in `feature_store` or `model_server` |
| ☠️ Cascade — Multi-Failure | Expert | Upstream `embedding_service_v3` root cause |
""")

    # ─ Event handlers
    def update_task_desc(task_id: str) -> str:
        desc = TASK_META.get(task_id, {}).get("desc", "")
        return f'<div class="task-desc">{desc}</div>'

    task_radio.change(
        fn=update_task_desc,
        inputs=task_radio,
        outputs=task_desc_box,
    )

    run_btn.click(
        fn=run_sentinel,
        inputs=task_radio,
        outputs=[log_box, thinking_box, diag_box, score_box, status_box],
        show_progress=True,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
