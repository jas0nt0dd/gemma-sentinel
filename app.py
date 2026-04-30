"""
SentinelAI — Gradio Demo App
Gemma 4 Good Hackathon | Global Resilience Impact Track

How it works:
  1. User picks a task scenario (easy / medium / hard / cascade)
  2. SentinelAI (Gemma via HF Inference API) reads the live MLOps env
  3. Agent investigates: queries logs, checks metrics, compares configs
  4. Gemma reasons and outputs a structured JSON diagnosis
  5. Diagnosis is submitted to the env -> real score returned

Env vars (set as HF Space secrets):
  HF_TOKEN        Your HuggingFace token (Inference API access)
  HF_SPACE_URL    MLOps Incident env Space URL
  MODEL_ID        Override model (default: google/gemma-2-2b-it)
"""

import json
import os
import re
import time

import gradio as gr
import requests
from huggingface_hub import InferenceClient

# --- Config ---
HF_TOKEN  = os.getenv("HF_TOKEN", "")
SPACE_URL = os.getenv("HF_SPACE_URL", "https://jason9150-mlops-incident-env.hf.space").rstrip("/")
MODEL_ID  = os.getenv("MODEL_ID", "google/gemma-2-2b-it")

# --- Task metadata ---
TASK_META = {
    "easy": {
        "label": "Easy - Data Quality Alert",
        "desc": "A data pipeline is causing accuracy drops. Schema migration went wrong.",
        "investigate": [
            ("query_logs",    "data_pipeline_a"),
            ("check_metrics", "data_pipeline_a"),
            ("check_metrics", "feature_store"),
        ],
    },
    "medium": {
        "label": "Medium - Latency Spike",
        "desc": "Model serving latency spiked after a config change. Bottleneck unknown.",
        "investigate": [
            ("inspect",         "feature_preprocessor_v2"),
            ("compare_configs", "feature_preprocessor_v2"),
            ("check_metrics",   "model_server"),
        ],
    },
    "hard": {
        "label": "Hard - Silent Model Drift",
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
        "label": "Cascade - Multi-System Failure",
        "desc": "3 simultaneous failures triggered by one bad deployment. Full cascade investigation.",
        "investigate": [
            ("check_metrics", "embedding_service_v3"),
            ("inspect",       "feature_store"),
            ("check_metrics", "ab_test_router"),
        ],
    },
}

SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent.
You diagnose production AI/ML system failures from component status and evidence.

Rules:
- Identify components in degraded, error, warning, or critical states.
- Weigh log evidence, metric spikes, config changes, and feature drift signals.
- For cascade failures, find the upstream root that explains all downstream symptoms.
- target must exactly match a component name from COMPONENT STATUS.

Output ONLY this single-line JSON (no markdown, no extra text):
{"target":"<exact_component_name>","root_cause":"<one sentence>","fix":"<one sentence>"}"""


# --- Environment helpers ---
def _post(endpoint, payload, retries=3):
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


def env_reset(task_id):
    return _post("reset", {"task_id": task_id})


def env_step(action, target):
    return _post("step", {"action_type": action, "target": target, "parameters": {}})


def env_health():
    try:
        return requests.get(f"{SPACE_URL}/health", timeout=8).status_code == 200
    except Exception:
        return False


def parse_json_from_text(text):
    try:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


# --- Main agent ---
def run_sentinel(task_id):
    log_lines = []
    diagnosis = {}
    score     = 0.0
    status    = "Starting..."

    def emit():
        log_text   = "\n".join(log_lines)
        diag_text  = json.dumps(diagnosis, indent=2) if diagnosis else ""
        score_html = _score_badge(score) if diagnosis else ""
        return log_text, diag_text, score_html, status

    # Step 1: health
    status = "Checking environment..."
    yield emit()
    if not env_health():
        status = "ERROR: Environment unreachable. Check HF_SPACE_URL secret."
        log_lines.append("Cannot reach: " + SPACE_URL)
        yield emit()
        return
    log_lines.append("Environment online: " + SPACE_URL)
    log_lines.append("Model: " + MODEL_ID)

    # Step 2: reset
    status = "Resetting scenario..."
    yield emit()
    try:
        obs = env_reset(task_id)
    except Exception as e:
        status = "Reset failed: " + str(e)
        log_lines.append("ERROR: " + str(e))
        yield emit()
        return

    alert   = obs.get("alert_summary", "(no alert)")
    goal    = obs.get("goal", "(no goal)")
    comp_st = obs.get("component_status", {}) or {}

    log_lines.append("")
    log_lines.append("ALERT: " + alert)
    log_lines.append("GOAL:  " + goal)
    log_lines.append("")
    log_lines.append("COMPONENT STATUS:")
    for k, v in comp_st.items():
        tag = "[ERR] " if v in ("error", "critical") else "[WRN] " if v == "degraded" else "[ OK] "
        log_lines.append("  " + tag + k + ": " + v)
    yield emit()

    # Step 3: investigate
    status = "Investigating..."
    evidence_lines = []
    for action, target in TASK_META[task_id]["investigate"]:
        log_lines.append("")
        log_lines.append("> " + action + "(" + target + ")")
        yield emit()
        try:
            s  = env_step(action, target)
            fb = (s.get("action_feedback") or "")[:500]
            if fb:
                log_lines.append("  " + fb)
                evidence_lines.append("[" + action + "(" + target + ")] " + fb)
            if s.get("done"):
                break
        except Exception as e:
            log_lines.append("  WARNING: " + str(e))
        time.sleep(0.1)
        yield emit()

    # Step 4: call model
    status = "Model reasoning..."
    yield emit()

    status_lines  = "\n".join("- " + k + ": " + v for k, v in comp_st.items())[:900]
    evidence_text = "\n".join(evidence_lines) or "(none collected)"
    user_msg = (
        "ALERT: " + alert + "\n"
        "GOAL: " + goal + "\n\n"
        "COMPONENT STATUS:\n" + status_lines + "\n\n"
        "EVIDENCE:\n" + evidence_text + "\n\n"
        "Output the JSON diagnosis now."
    )

    raw_output = ""
    try:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not set as a Space secret.")

        client = InferenceClient(
            model=MODEL_ID,
            token=HF_TOKEN,
            provider="hf-inference",
        )
        messages = [{"role": "user", "content": SYS_PROMPT + "\n\n" + user_msg}]

        for chunk in client.chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.2,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            raw_output += delta
            yield emit()

    except Exception as e:
        err = str(e)
        status = "Model call failed."
        log_lines.append("")
        log_lines.append("ERROR: " + err)
        if "not supported" in err:
            log_lines.append("")
            log_lines.append("FIX: Go to Space Settings and set MODEL_ID to one of:")
            log_lines.append("  - google/gemma-2-2b-it")
            log_lines.append("  - mistralai/Mistral-7B-Instruct-v0.3")
            log_lines.append("  - meta-llama/Llama-3.2-3B-Instruct")
        elif "401" in err or "403" in err:
            log_lines.append("FIX: HF_TOKEN is invalid or has no Inference API access.")
        yield emit()
        return

    log_lines.append("")
    log_lines.append("--- Model raw output ---")
    log_lines.append(raw_output[:600])
    yield emit()

    # Step 5: parse
    status = "Parsing diagnosis..."
    parsed = parse_json_from_text(raw_output)
    if not parsed or "target" not in parsed:
        status = "Could not parse JSON from model output. Trying fallback..."
        log_lines.append("WARNING: No valid JSON found. Using evidence-based fallback.")
        # Fallback: pick highest-severity component
        fallback_target = next(
            (k for k, v in comp_st.items() if v in ("error", "critical")),
            next((k for k, v in comp_st.items() if v == "degraded"), None)
        )
        if fallback_target:
            parsed = {
                "target": fallback_target,
                "root_cause": "Highest severity component identified by fallback heuristic.",
                "fix": "Investigate " + fallback_target + " logs and metrics."
            }
            log_lines.append("Fallback target: " + fallback_target)
        else:
            yield emit()
            return

    diagnosis = parsed
    log_lines.append("")
    log_lines.append("DIAGNOSIS -> target: " + parsed["target"])
    log_lines.append("Root cause: " + parsed.get("root_cause", ""))
    log_lines.append("Fix: " + parsed.get("fix", ""))
    yield emit()

    # Step 6: submit
    status = "Submitting diagnosis..."
    yield emit()
    try:
        result = env_step("submit_diagnosis", parsed["target"])
        score  = float(result.get("final_score") or result.get("reward") or 0.0)
        fb     = result.get("action_feedback") or ""
        log_lines.append("")
        log_lines.append("SCORE: " + str(round(score, 4)))
        if fb:
            log_lines.append("Feedback: " + fb)
    except Exception as e:
        log_lines.append("WARNING: Submission failed: " + str(e))
        score = 0.0

    status = _status_from_score(score)
    yield emit()


def _score_badge(score):
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
        '<div style="display:inline-block;background:' + color
        + ';color:white;padding:8px 24px;border-radius:8px;'
        + 'font-size:1.5rem;font-weight:700;margin-top:8px;">'
        + str(pct) + '% &mdash; ' + label + '</div>'
    )


def _status_from_score(score):
    if score >= 0.9:
        return "Perfect diagnosis! SentinelAI nailed it."
    elif score >= 0.7:
        return "Good diagnosis - correct root identified."
    elif score >= 0.4:
        return "Partial credit - partially correct."
    elif score > 0.0:
        return "Low score - wrong component targeted."
    else:
        return "Miss - incorrect diagnosis."


# --- Gradio UI ---
TASK_CHOICES = [(v["label"], k) for k, v in TASK_META.items()]

CSS = """
.sentinel-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2d4a 100%);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 16px;
    border: 1px solid #334155;
}
.sentinel-header h1 { color: #38bdf8; font-size: 2rem; font-weight: 800; margin: 0 0 4px 0; }
.sentinel-header p  { color: #94a3b8; margin: 0; font-size: 0.95rem; }
.task-desc {
    background: #1e293b; border-left: 4px solid #38bdf8;
    border-radius: 0 8px 8px 0; padding: 10px 16px;
    color: #cbd5e1; font-size: 0.9rem; margin-top: 8px;
}
"""

with gr.Blocks(
    title="SentinelAI - MLOps Incident Response Agent",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="sky",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as demo:

    gr.HTML("""
    <div class="sentinel-header">
        <h1>SentinelAI</h1>
        <p>Autonomous MLOps Incident Response &middot; Powered by Gemma &middot; Gemma 4 Good Hackathon</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            task_radio = gr.Radio(
                choices=TASK_CHOICES,
                value="easy",
                label="Select Incident Scenario",
                info="Each scenario tests different failure patterns.",
            )
            task_desc_box = gr.HTML(
                value='<div class="task-desc">' + TASK_META["easy"]["desc"] + '</div>'
            )
            run_btn = gr.Button("Run SentinelAI", variant="primary", size="lg")
            gr.Markdown("""
---
**How it works:**
1. Resets the live RL environment
2. Investigates: logs, metrics, drift signals
3. Model reads all evidence and reasons
4. Outputs structured JSON diagnosis
5. Real score returned from environment

**Env:** [MLOps Incident OpenEnv](https://huggingface.co/spaces/jason9150/mlops-incident-env)
""")

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Status",
                value="Select a scenario and click Run SentinelAI.",
                interactive=False, max_lines=1,
            )
            score_box = gr.HTML()

            with gr.Tabs():
                with gr.TabItem("Investigation Log"):
                    log_box = gr.Textbox(
                        label="", lines=20, max_lines=35,
                        interactive=False,
                        placeholder="Investigation log will appear here...",
                    )
                with gr.TabItem("Diagnosis JSON"):
                    diag_box = gr.Code(
                        label="", language="json",
                        lines=8, interactive=False,
                    )

    gr.Markdown("""
---
### Scenario Guide
| Scenario | Difficulty | Root cause location |
|---|---|---|
| Easy - Data Quality | Beginner | `data_pipeline_a` schema change |
| Medium - Latency Spike | Intermediate | `feature_preprocessor_v2` config |
| Hard - Silent Drift | Advanced | `feature_store` PSI spike |
| Cascade - Multi-Failure | Expert | `embedding_service_v3` upstream |
""")

    def update_desc(task_id):
        desc = TASK_META.get(task_id, {}).get("desc", "")
        return '<div class="task-desc">' + desc + '</div>'

    task_radio.change(fn=update_desc, inputs=task_radio, outputs=task_desc_box)

    run_btn.click(
        fn=run_sentinel,
        inputs=task_radio,
        outputs=[log_box, diag_box, score_box, status_box],
        show_progress=True,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
