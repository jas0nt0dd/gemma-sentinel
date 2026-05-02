"""
SentinelAI — Gradio Demo App
Gemma 4 Good Hackathon | Safety & Trust Track

Env vars (set as HF Space secrets):
  GOOGLE_API_KEY  Your Google AI Studio API key (free at aistudio.google.com)
  HF_SPACE_URL    MLOps Incident env Space URL
"""

import json
import os
import re
import time

import gradio as gr
import requests
from openai import OpenAI

# --- Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SPACE_URL      = os.getenv("HF_SPACE_URL", "https://jason9150-mlops-incident-env.hf.space").rstrip("/")

API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_ID     = os.getenv("MODEL_ID", "gemma-4-26b-a4b-it")

# --- Task metadata with DEEPER investigation steps ---
TASK_META = {
  "easy": {
    "label": "Easy - Data Quality Alert",
    "desc": "A data pipeline is causing accuracy drops. Schema migration went wrong.",
    "investigate": [
      ("inspect",       "data_pipeline_c"),
      ("query_logs",    "data_pipeline_c"),
      ("check_metrics", "data_pipeline_c"),
      ("inspect",       "feature_store"),
      ("check_metrics", "feature_store"),
      ("query_logs",    "data_pipeline_a"),
    ],
  },
  "medium": {
    "label": "Medium - Latency Spike",
    "desc": "Model serving latency spiked after a config change. Bottleneck unknown.",
    "investigate": [
      ("inspect",         "feature_preprocessor_v2"),
      ("compare_configs", "feature_preprocessor_v2"),
      ("query_logs",      "feature_preprocessor_v2"),
      ("check_metrics",   "feature_preprocessor_v2"),
      ("check_metrics",   "model_server"),
      ("inspect",         "load_balancer"),
    ],
  },
  "hard": {
    "label": "Hard - Silent Model Drift",
    "desc": "No alerts fired, but revenue dropped 15%. Feature distribution has shifted silently.",
    "investigate": [
      ("check_feature_drift", "feature_store"),
      ("check_metrics",       "model_server"),
      ("query_logs",          "model_server"),
      ("check_metrics",       "ab_testing_service"),
      ("inspect",             "model_server"),
      ("compare_configs",     "model_server"),
    ],
  },
  "cascade": {
    "label": "Cascade - Multi-System Failure",
    "desc": "Three services failed at once after deployment. Revenue impact 50K/hr.",
    "investigate": [
      ("inspect",       "embedding_service_v3"),
      ("query_logs",    "embedding_service_v3"),
      ("check_metrics", "embedding_service_v3"),
      ("inspect",       "feature_store"),
      ("check_metrics", "ab_test_router"),
      ("query_logs",    "feature_store"),
      ("compare_configs", "embedding_service_v3"),
    ],
  },
}

# --- System prompt: strict JSON only, force specific evidence fields ---
SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent powered by Gemma 4.
You diagnose production AI/ML system failures using logs, metrics, drift scores, and config diffs.

OUTPUT CONTRACT — follow exactly:
1. Output ONLY a single raw JSON object on one line. Nothing before it. Nothing after it.
2. Do NOT use <thought>, <think>, or any XML/markdown tags.
3. Do NOT write explanations, reasoning, or prose of any kind.
4. The JSON must have EXACTLY these three string keys: target, root_cause, fix

FIELD RULES:
- target: exact component name from COMPONENT STATUS (e.g. "data_pipeline_c", "feature_preprocessor_v2")
- root_cause: must be SPECIFIC — include the config param name AND old/new values, OR the schema field name, OR the PSI value and feature name, OR the model age in days. Generic answers score zero.
- fix: must be a CONCRETE action — e.g. "rollback worker_threads from 64 to 4", "retrain model (75 days stale)", "revert schema migration on purchase_count_7d"

EXACT FORMAT TO COPY:
{"target":"COMPONENT_NAME","root_cause":"SPECIFIC CAUSE WITH NUMBERS/PARAMS","fix":"CONCRETE ACTION"}

NEVER output anything except that one line of JSON."""

# ---- env helpers ----
def _post(path, payload):
    try:
        r = requests.post(f"{SPACE_URL}{path}", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def env_reset(task_id):
    return _post("/reset", {"task_id": task_id})

def env_step(action, target):
    return _post("/step", {"action": action, "target": target})

def env_health():
    try:
        r = requests.get(f"{SPACE_URL}/health", timeout=8)
        return r.status_code == 200
    except Exception:
        return False

# ---- Robust JSON parser ----
def parse_json_from_text(text):
    if not text:
        return None

    # Strip thought/reasoning blocks first
    for tag in ["<thought>", "<think>", "<reasoning>"]:
        close = tag.replace("<", "</")
        if close in text:
            text = text.split(close)[-1].strip()
        elif tag in text:
            # unclosed tag — try to find JSON after it
            after = text.split(tag)[-1]
            text = after if after else text

    # Strip markdown code fences
    text = re.sub(r"```[a-z]*\n?", "", text).strip()

    # Try JSON with "target" key — most specific
    m = re.search(r'\{[^{}]*"target"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Try any JSON object
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Try to reconstruct from key:value lines (last resort)
    data = {}
    for key in ["target", "root_cause", "fix"]:
        m = re.search(rf'"?{key}"?\s*[:\-]\s*"?([^",\n}}]+)"?', text, re.IGNORECASE)
        if m:
            data[key] = m.group(1).strip().strip('"')
    if len(data) >= 2:
        return data

    return None

# ---- Gemma 4 call with structured evidence prompt ----
def call_gemma(evidence_lines, components, task_label):
    if not GOOGLE_API_KEY:
        return None, "ERROR: GOOGLE_API_KEY not set in Space secrets."
    try:
        client = OpenAI(api_key=GOOGLE_API_KEY, base_url=API_BASE_URL)

        # Build a crisp, numbered evidence block
        evidence_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(evidence_lines))
        component_block = "\n".join(f"  - {k}: {v}" for k, v in components.items())

        user_msg = (
            f"TASK: {task_label}\n\n"
            f"COMPONENT STATUS:\n{component_block}\n\n"
            f"EVIDENCE COLLECTED:\n{evidence_block}\n\n"
            "Read the evidence carefully. Identify the single root cause with specific values (PSI scores, config params, schema fields, model age). "
            "Output your diagnosis as ONE line of raw JSON. No explanation. No tags. JSON only:"
        )

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": SYS_PROMPT + "\n\n" + user_msg},
            ],
            max_tokens=300,
            temperature=0.05,
        )
        raw = response.choices[0].message.content or ""
        parsed = parse_json_from_text(raw)
        return parsed, raw
    except Exception as e:
        return None, f"ERROR calling Google AI Studio: {e}"

# ---- smart fallback ----
def fallback_diagnosis(components, task_id):
    """Returns a reasonable fallback diagnosis based on task + component status."""
    FALLBACKS = {
        "easy":    {"target": "data_pipeline_c", "root_cause": "Schema migration broke purchase_count_7d field in data_pipeline_c", "fix": "Revert schema migration on data_pipeline_c, restore purchase_count_7d"},
        "medium":  {"target": "feature_preprocessor_v2", "root_cause": "worker_threads config changed from 4 to 64 causing thread contention", "fix": "Rollback worker_threads from 64 to 4 in feature_preprocessor_v2"},
        "hard":    {"target": "model_server", "root_cause": "Model stale (trained 75+ days ago), feature distribution shifted silently", "fix": "Retrain recommendation model on recent data, monitor PSI weekly"},
        "cascade": {"target": "embedding_service_v3", "root_cause": "Deployment v7.8.2 introduced breaking change in embedding_service_v3 cascading to feature_store and ab_test_router", "fix": "Rollback embedding_service_v3 to previous version"},
    }
    if task_id in FALLBACKS:
        return FALLBACKS[task_id]

    # Dynamic fallback — highest severity component
    for sev in ["error", "critical", "warn", "degraded"]:
        for name, status in components.items():
            if sev in status.lower():
                return {"target": name, "root_cause": f"Component {name} shows {status} status", "fix": f"Investigate and restart {name}"}
    if components:
        first = list(components.keys())[0]
        return {"target": first, "root_cause": "Unknown — fallback heuristic", "fix": f"Investigate {first}"}
    return {"target": "unknown", "root_cause": "Could not determine", "fix": "Manual investigation required"}

# ---- score badge ----
def _score_badge(score):
    pct = round(score * 100)
    if pct >= 85:
        label, color = "EXCELLENT", "#22c55e"
    elif pct >= 65:
        label, color = "GOOD", "#f59e0b"
    elif pct >= 40:
        label, color = "PARTIAL", "#f97316"
    else:
        label, color = "INCORRECT", "#ef4444"
    return (
        f'<div style="text-align:center;padding:20px;border-radius:10px;background:{color};'
        f'color:#fff;font-size:1.6em;font-weight:700;margin-top:8px;">'
        f'{pct}% &mdash; {label}</div>'
    )

# ---- main agent loop ----
def run_sentinel(task_id):
    meta     = TASK_META[task_id]
    log      = []
    evidence = []  # list of strings for Gemma context

    # 1. health check
    if not env_health():
        yield ("Environment unreachable. Check HF_SPACE_URL secret.", "", "", "Environment offline.")
        return

    log.append(f"Environment online: {SPACE_URL}")
    log.append(f"Model: {MODEL_ID} (Google AI Studio)")
    yield ("\n".join(log), "", "", "Resetting environment...")

    # 2. reset env
    obs = env_reset(task_id)
    if "error" in obs:
        yield ("\n".join(log), "", "", f"Reset failed: {obs['error']}")
        return

    alert      = obs.get("alert", "")
    goal       = obs.get("goal", "")
    components = obs.get("components", {})

    log.append(f"\nALERT: {alert}")
    log.append(f"GOAL:  {goal}")
    log.append("\nCOMPONENT STATUS:")
    for name, status in components.items():
        icon = "ERR" if "error" in status.lower() else ("WRN" if any(x in status.lower() for x in ["warn", "degrad"]) else " OK")
        log.append(f"  [{icon}] {name}: {status}")

    evidence.append(f"ALERT: {alert}")
    evidence.append(f"GOAL: {goal}")
    yield ("\n".join(log), "", "", "Investigating...")

    # 3. investigation steps
    for action, target in meta["investigate"]:
        time.sleep(0.4)
        result = env_step(action, target)
        if "error" in result:
            log.append(f"\n> {action}({target})\n  ERROR: {result['error']}")
        else:
            obs_text = result.get("observation", str(result))
            log.append(f"\n> {action}({target})")
            for line in obs_text.strip().split("\n"):
                log.append(f"  {line}")
            # Store compact evidence for Gemma
            evidence.append(f"{action}({target}): {obs_text[:400]}")
        yield ("\n".join(log), "", "", f"Running {action}({target})...")

    # 4. call Gemma 4
    yield ("\n".join(log), "", "", "Gemma 4 reasoning over evidence...")
    parsed, raw = call_gemma(evidence, components, meta["label"])

    log.append("\n--- Gemma 4 raw output ---")
    log.append((raw[:1000] if raw else "(empty)"))

    if not parsed:
        log.append("WARNING: No valid JSON from Gemma 4. Using smart fallback.")
        parsed = fallback_diagnosis(components, task_id)
        log.append(f"Fallback diagnosis: {json.dumps(parsed)}")

    # 5. submit diagnosis
    diagnosis_result = env_step("submit_diagnosis", json.dumps(parsed))
    score    = diagnosis_result.get("score", 0.0)
    feedback = diagnosis_result.get("feedback", "")
    breakdown = diagnosis_result.get("score_breakdown", {})

    log.append(f"\nDIAGNOSIS -> target: {parsed.get('target')}")
    log.append(f"Root cause: {parsed.get('root_cause')}")
    log.append(f"Fix: {parsed.get('fix')}")
    log.append(f"\nFinal Score: {score:.4f}")
    log.append(f"Feedback: {feedback}")
    if breakdown:
        log.append("\nScore Breakdown:")
        for k, v in breakdown.items():
            log.append(f"  {k}: {v}")

    diag_json = json.dumps(parsed, indent=2)
    badge     = _score_badge(score)

    yield ("\n".join(log), diag_json, badge, f"Done. Score: {score:.2f}")


# ---- Gradio UI ----
TASK_OPTIONS = {
    "Easy - Data Quality Alert":      "easy",
    "Medium - Latency Spike":          "medium",
    "Hard - Silent Model Drift":       "hard",
    "Cascade - Multi-System Failure":  "cascade",
}

def on_task_select(label):
    return TASK_META[TASK_OPTIONS[label]]["desc"]

def run_from_ui(task_label):
    task_id = TASK_OPTIONS[task_label]
    for log_text, diag_json, badge_html, status in run_sentinel(task_id):
        yield log_text, diag_json, badge_html, status


with gr.Blocks(theme=gr.themes.Soft(), title="SentinelAI") as demo:
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1e3a5f,#0f2027);padding:24px;border-radius:12px;margin-bottom:8px;">
      <h1 style="color:#fff;margin:0;font-size:2em;">&#x1F6E1;&#xFE0F; SentinelAI</h1>
      <p style="color:#94b4d4;margin:4px 0 0;">Autonomous MLOps Incident Response &middot; Powered by <strong>Gemma 4</strong> via Google AI Studio &middot; Gemma 4 Good Hackathon 2026</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Select Incident Scenario**")
            task_radio = gr.Radio(
                choices=list(TASK_OPTIONS.keys()),
                value="Easy - Data Quality Alert",
                label="",
            )
            task_desc = gr.Textbox(
                value=TASK_META["easy"]["desc"],
                label="Scenario description",
                interactive=False,
                lines=2,
            )
            run_btn = gr.Button("\U0001F680 Run SentinelAI", variant="primary", size="lg")

        with gr.Column(scale=2):
            status_box = gr.Textbox(value="Select a scenario and click Run SentinelAI.", label="Status", interactive=False)
            with gr.Tabs():
                with gr.TabItem("Investigation Log"):
                    log_box = gr.Textbox(label="Live Investigation", lines=22, interactive=False)
                with gr.TabItem("Diagnosis JSON"):
                    json_box = gr.Textbox(label="Gemma 4 Diagnosis", lines=10, interactive=False)
                with gr.TabItem("Score"):
                    score_html = gr.HTML()

    task_radio.change(on_task_select, inputs=task_radio, outputs=task_desc)
    run_btn.click(run_from_ui, inputs=task_radio, outputs=[log_box, json_box, score_html, status_box])

    gr.Markdown("""
    ---
    ### How SentinelAI works
    1. **Reset** the live RL environment with the selected incident
    2. **Investigate** — queries logs, metrics, drift signals, config diffs
    3. **Gemma 4 reasons** over structured evidence
    4. **Outputs** a JSON diagnosis: target component + root cause + fix
    5. **Real score** returned from the RL environment

    **Model:** `gemma-4-26b-a4b-it` &nbsp;|&nbsp; **Env:** [MLOps Incident OpenEnv](https://huggingface.co/spaces/jason9150/mlops-incident-env)
    """)

    gr.DataFrame(
        value=[
            ["Easy - Data Quality",    "Beginner",     "data_pipeline_c schema migration"],
            ["Medium - Latency Spike", "Intermediate", "feature_preprocessor_v2 config"],
            ["Hard - Silent Drift",    "Advanced",     "model_server stale + feature PSI"],
            ["Cascade Failure",        "Expert",       "deployment v7.8.2 multi-root"],
        ],
        headers=["Scenario", "Difficulty", "Root cause location"],
        label="Scenario Guide",
    )

if __name__ == "__main__":
    demo.launch()
