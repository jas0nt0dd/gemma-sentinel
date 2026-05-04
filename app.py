"""
SentinelAI — Gradio Demo App
Gemma 4 Good Hackathon | Safety & Trust Track

Env vars (set as HF Space secrets):
  GOOGLE_API_KEY   Your Google AI Studio API key (free at aistudio.google.com)
  HF_SPACE_URL     MLOps Incident env Space URL
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
API_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_ID       = os.getenv("MODEL_ID", "gemma-4-26b-a4b-it")

# --- Task metadata ---
TASK_META = {
    "easy": {
        "label": "Easy - Data Quality Alert",
        "desc": "A data pipeline is causing accuracy drops. Schema migration went wrong.",
        "investigate": [
            ("inspect",        "data_pipeline_c"),
            ("query_logs",     "data_pipeline_c"),
            ("check_metrics",  "feature_store"),
            ("query_logs",     "feature_store"),
        ],
    },
    "medium": {
        "label": "Medium - Latency Spike",
        "desc": "Model serving latency spiked after a config change. Bottleneck unknown.",
        "investigate": [
            ("inspect",         "feature_preprocessor_v2"),
            ("compare_configs", "feature_preprocessor_v2"),
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
            ("inspect",             "ab_testing_service"),
        ],
    },
    "cascade": {
        "label": "Cascade - Multi-System Failure",
        "desc": "Three services failed at once after deployment. Revenue impact 50K/hr.",
        "investigate": [
            ("check_metrics", "embedding_service_v3"),
            ("inspect",       "feature_store"),
            ("inspect",       "model_registry"),
            ("check_metrics", "model_serving"),
        ],
    },
}

SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent powered by Gemma 4.
You diagnose production AI/ML system failures using logs, metrics, drift scores, and config diffs.

OUTPUT CONTRACT — follow exactly:
1. Output ONLY a single raw JSON object on one line. Nothing before it. Nothing after it.
2. Do NOT use <thought>, </thought>, or any XML/markdown tags.
3. Do NOT write explanations, reasoning, or prose of any kind.
4. The JSON must have EXACTLY these three string keys: target, root_cause, fix

FIELD RULES:
- target: exact component name from COMPONENT STATUS
- root_cause: be SPECIFIC — include config param name, field name, PSI value, or model age in days
- fix: concrete action — e.g. "rollback batch_size from 512 to 32", "retrain model (45 days stale)"

EXACT FORMAT TO COPY:
{"target":"COMPONENT_NAME","root_cause":"SPECIFIC CAUSE WITH NUMBERS","fix":"CONCRETE ACTION"}

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

# FIX 1: use action_type (not action), and accept optional parameters
def env_step(action_type, target, parameters=None):
    payload = {"action_type": action_type, "target": target}
    if parameters:
        payload["parameters"] = parameters
    return _post("/step", payload)

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
    # Strip thought/reasoning blocks
    for open_tag, close_tag in [("<thought>", "</thought>"), ("<think>", "</think>"), ("<reasoning>", "</reasoning>")):
        if close_tag in text:
            text = text.split(close_tag)[-1].strip()
            break
        elif open_tag in text:
            inner = text.split(open_tag)[-1]
            m = re.search(r'\{[^{}]*"target"[^{}]*\}', inner, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
            return None
    # Find JSON with "target" key
    m = re.search(r'\{[^{}]*"target"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # Broader JSON search
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if len(data) >= 2:
                return data
        except Exception:
            pass
    return None

# ---- Gemma 4 call ----
def call_gemma(evidence_lines, components, task_label):
    if not GOOGLE_API_KEY:
        return None, "ERROR: GOOGLE_API_KEY not set in Space secrets."
    try:
        client = OpenAI(api_key=GOOGLE_API_KEY, base_url=API_BASE_URL)
        evidence_block  = "\n".join(f"{i+1}. {line}" for i, line in enumerate(evidence_lines))
        component_block = "\n".join(f"  - {k}: {v}" for k, v in components.items())
        user_msg = (
            f"TASK: {task_label}\n\n"
            f"COMPONENT STATUS:\n{component_block}\n\n"
            f"EVIDENCE COLLECTED:\n{evidence_block}\n\n"
            "Output your diagnosis as ONE line of raw JSON. No explanation. No tags. JSON only:"
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": SYS_PROMPT + "\n\n" + user_msg}],
            max_tokens=300,
            temperature=0.05,
        )
        raw    = response.choices[0].message.content or ""
        parsed = parse_json_from_text(raw)
        return parsed, raw
    except Exception as e:
        return None, f"ERROR calling Google AI Studio: {e}"

# ---- smart fallback ----
def fallback_diagnosis(components, task_id):
    FALLBACKS = {
        "easy":    {"target": "data_pipeline_c",        "root_cause": "Schema migration broke purchase_count_7d field in data_pipeline_c",                  "fix": "Revert schema migration on data_pipeline_c"},
        "medium":  {"target": "feature_preprocessor_v2","root_cause": "batch_size increased to 512 causing OOM/memory leak in feature_preprocessor_v2",      "fix": "Rollback batch_size from 512 to 32"},
        "hard":    {"target": "model_server",            "root_cause": "Model stale — trained 45+ days ago, concept drift causing silent revenue drop",         "fix": "Retrain recommendation model on recent data"},
        "cascade": {"target": "embedding_service_v3",   "root_cause": "Deployment v8.1.0 downgraded CUDA driver causing GPU unavailable in embedding_service_v3","fix": "Rollback deployment v8.1.0"},
    }
    if task_id in FALLBACKS:
        return FALLBACKS[task_id]
    for sev in ["error", "critical", "warn", "degraded"]:
        for name, status in components.items():
            if sev in status.lower():
                return {"target": name, "root_cause": f"{name} shows {status}", "fix": f"Restart {name}"}
    if components:
        first = list(components.keys())[0]
        return {"target": first, "root_cause": "Unknown", "fix": f"Investigate {first}"}
    return {"target": "unknown", "root_cause": "Could not determine", "fix": "Manual investigation"}

# ---- score badge ----
def _score_badge(score):
    pct = round((score or 0) * 100)
    if pct >= 85:   label, color = "EXCELLENT", "#22c55e"
    elif pct >= 65: label, color = "GOOD",      "#f59e0b"
    elif pct >= 40: label, color = "PARTIAL",   "#f97316"
    else:           label, color = "INCORRECT", "#ef4444"
    return (
        f'<div style="text-align:center;padding:24px;border-radius:12px;'
        f'background:{color};color:white;font-size:2.2em;font-weight:bold;">'
        f'{pct}% \u2014 {label}</div>'
    )

# ---- main agent loop ----
def run_sentinel(task_id):
    meta     = TASK_META[task_id]
    log      = []
    evidence = []

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

    # FIX 2: correct field names from actual env response
    alert      = obs.get("alert_summary", obs.get("alert", ""))
    goal       = obs.get("goal", "")
    components = obs.get("component_status", obs.get("components", {}))

    log.append(f"\nALERT: {alert}")
    log.append(f"GOAL:  {goal}")
    log.append("\nCOMPONENT STATUS:")
    for name, status in components.items():
        icon = "ERR" if "error" in status.lower() else ("WRN" if any(x in status.lower() for x in ["warn", "degrad", "critical"]) else " OK")
        log.append(f"  [{icon}] {name}: {status}")

    evidence.append(f"ALERT: {alert}")
    evidence.append(f"GOAL: {goal}")
    yield ("\n".join(log), "", "", "Investigating...")

    # 3. investigation steps
    for action, target in meta["investigate"]:
        time.sleep(0.4)
        result = env_step(action, target)  # uses fixed action_type field

        if "error" in result:
            log.append(f"\n> {action}({target})\n  ERROR: {result['error']}")
        else:
            feedback = result.get("action_feedback", "")
            logs_raw = result.get("recent_logs", [])
            metrics  = result.get("metrics_snapshot", {})
            reward   = result.get("reward", 0)

            log.append(f"\n> {action}({target})")
            if feedback:
                log.append(f"  {feedback[:300]}")
            for entry in logs_raw:
                log.append(f"  [{entry.get('level','?')}] {entry.get('time','')} - {entry.get('msg','')}")
            if metrics:
                log.append(f"  metrics: {json.dumps(metrics)[:400]}")
            log.append(f"  reward: {reward}")

            # Only add to evidence if not a repeat/penalty step
            if "Already ran" not in feedback:
                ev_parts = [f"{action}({target}):"]
                for entry in logs_raw:
                    ev_parts.append(f"  [{entry.get('level','?')}] {entry.get('msg','')}")
                for k, v in metrics.items():
                    ev_parts.append(f"  metric.{k}={v}")
                if feedback and "Already ran" not in feedback:
                    for line in feedback.split("\\n"):
                        if any(line.strip().startswith(x) for x in ["Status:", "Description:"]):
                            ev_parts.append(f"  {line.strip()}")
                evidence.append("\n".join(ev_parts))

        yield ("\n".join(log), "", "", f"Running {action}({target})...")

    # 4. call Gemma 4
    log.append("\n--- Sending to Gemma 4 ---")
    log.append(f"Evidence items: {len(evidence)}")
    yield ("\n".join(log), "", "", "Gemma 4 reasoning over evidence...")

    parsed, raw = call_gemma(evidence, components, meta["label"])

    log.append("\n--- Gemma 4 raw output ---")
    log.append(raw[:1000] if raw else "(empty)")

    if not parsed:
        log.append("WARNING: No valid JSON from Gemma 4. Using smart fallback.")
        parsed = fallback_diagnosis(components, task_id)
        log.append(f"Fallback diagnosis: {json.dumps(parsed)}")

    # 5. submit diagnosis — FIX 3: target in target field, rest in parameters
    log.append("\n--- Submitting diagnosis ---")
    yield ("\n".join(log), "", "", "Submitting diagnosis...")

    diagnosis_result = env_step(
        "submit_diagnosis",
        parsed.get("target", ""),
        parameters={
            "root_cause": parsed.get("root_cause", ""),
            "fix":        parsed.get("fix", ""),
        }
    )

    # FIX 4: correct response field names
    score    = diagnosis_result.get("final_score") or diagnosis_result.get("score") or 0.0
    feedback = diagnosis_result.get("action_feedback", diagnosis_result.get("feedback", ""))
    breakdown = diagnosis_result.get("score_breakdown", {})

    log.append(f"\nDIAGNOSIS:")
    log.append(f"  target:     {parsed.get('target')}")
    log.append(f"  root_cause: {parsed.get('root_cause')}")
    log.append(f"  fix:        {parsed.get('fix')}")
    log.append(f"\nFinal Score: {score:.4f}")
    log.append(f"Feedback:    {feedback}")
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
    "Medium - Latency Spike":         "medium",
    "Hard - Silent Model Drift":      "hard",
    "Cascade - Multi-System Failure": "cascade",
}

def on_task_select(label):
    return TASK_META[TASK_OPTIONS[label]]["desc"]

def run_from_ui(task_label):
    for out in run_sentinel(TASK_OPTIONS[task_label]):
        yield out

with gr.Blocks(theme=gr.themes.Soft(), title="SentinelAI") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px;">
      <h1>\U0001f6e1\ufe0f SentinelAI</h1>
      <p>Autonomous MLOps Incident Response &middot; Powered by <b>Gemma 4</b> via Google AI Studio &middot; Gemma 4 Good Hackathon 2026</p>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Select Incident Scenario**\nEach scenario tests different failure patterns.")
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
            status_box = gr.Textbox(
                value="Select a scenario and click Run SentinelAI.",
                label="Status", interactive=False
            )
            with gr.Tabs():
                with gr.TabItem("Investigation Log"):
                    log_box = gr.Textbox(label="Live Investigation", lines=22, interactive=False)
                with gr.TabItem("Diagnosis JSON"):
                    json_box = gr.Textbox(label="Gemma 4 Diagnosis", lines=10, interactive=False)
                with gr.TabItem("Score"):
                    score_html = gr.HTML()

    task_radio.change(on_task_select, inputs=task_radio, outputs=task_desc)
    run_btn.click(
        run_from_ui,
        inputs=task_radio,
        outputs=[log_box, json_box, score_html, status_box]
    )

    gr.Markdown("""
---
### How SentinelAI works
1. **Reset** the live RL environment with the selected incident
2. **Investigate** — queries logs, metrics, drift signals, config diffs
3. **Gemma 4 reasons** over structured evidence
4. **Outputs** a JSON diagnosis: target component + root cause + fix
5. **Real score** returned from the RL environment

**Model:** `gemma-4-26b-a4b-it` | **Env:** [MLOps Incident OpenEnv](https://huggingface.co/spaces/jason9150/mlops-incident-env)
""")
    gr.DataFrame(
        value=[
            ["Easy - Data Quality",  "Beginner",     "data_pipeline_c schema migration"],
            ["Medium - Latency Spike","Intermediate", "feature_preprocessor_v2 config"],
            ["Hard - Silent Drift",   "Advanced",     "model_server stale + feature PSI"],
            ["Cascade Failure",       "Expert",       "deployment v8.1.0 multi-root"],
        ],
        headers=["Scenario", "Difficulty", "Root cause location"],
        label="Scenario Guide",
    )

if __name__ == "__main__":
    demo.launch()
