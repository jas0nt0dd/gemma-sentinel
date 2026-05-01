"""
SentinelAI — Gradio Demo App
Gemma 4 Good Hackathon | Global Resilience Impact Track

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

# Uses Gemma 4 via Google's own API
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_ID     = os.getenv("MODEL_ID", "gemma-4-26b-a4b-it")

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
      ("check_metrics",       "business"),
      ("check_metrics",       "model_server"),
      ("compare_configs",     "model_server"),
    ],
  },
  "cascade": {
    "label": "Cascade - Multi-System Failure",
    "desc": "Three services failed at once after deployment. Revenue impact 50K/hr.",
    "investigate": [
      ("check_metrics", "embedding_service_v3"),
      ("inspect",       "feature_store"),
      ("check_metrics", "ab_test_router"),
      ("query_logs",    "embedding_service_v3"),
    ],
  },
}

SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent.
You diagnose production AI/ML system failures from logs, metrics, and config evidence.

STRICT OUTPUT RULES:
- Do NOT use <thought>, <think>, or any XML tags whatsoever
- Do NOT explain your reasoning in prose
- Output ONLY a single line of raw JSON, nothing else before or after
- JSON must have exactly these three keys: target, root_cause, fix

DIAGNOSIS RULES:
- target must exactly match a component name from COMPONENT STATUS
- For cascade failures identify the single upstream root that caused all downstream symptoms
- root_cause must be specific: mention config param name, schema field, PSI value, or model age
- fix must be a concrete action: rollback version, restart service, retrain model, revert config param

EXACT OUTPUT FORMAT — copy this structure and fill in the values:
{"target":"COMPONENT_NAME","root_cause":"SPECIFIC_CAUSE_HERE","fix":"CONCRETE_ACTION_HERE"}"""

# ---- env helpers ----
def _post(path, payload):
    try:
        r = requests.post(f"{SPACE_URL}{path}", json=payload, timeout=15)
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

# ---- JSON parser ----
def parse_json_from_text(text):
    # Step 1: strip thought blocks — get text AFTER </thought>
    if "</thought>" in text:
        text = text.split("</thought>")[-1].strip()
    elif "<thought>" in text:
        # thought block never closed — try JSON inside it
        thought = text.split("<thought>")[-1]
        m = re.search(r'\{[^{}]*"target"[^{}]*\}', thought, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None

    # Step 2: find JSON with "target" key
    m = re.search(r'\{[^{}]*"target"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Step 3: broader JSON search
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None

# ---- Gemma 4 call ----
def call_gemma(evidence_text, task_label):
    if not GOOGLE_API_KEY:
        return None, "ERROR: GOOGLE_API_KEY not set in Space secrets."
    try:
        client = OpenAI(api_key=GOOGLE_API_KEY, base_url=API_BASE_URL)
        user_msg = (
            f"TASK: {task_label}\n\n"
            f"EVIDENCE COLLECTED:\n{evidence_text}\n\n"
            "Based on the evidence above, output your diagnosis as a single line of JSON now. "
            "No explanation. No tags. JSON only:\n"
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": SYS_PROMPT + "\n\n" + user_msg},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        parsed = parse_json_from_text(raw)
        return parsed, raw
    except Exception as e:
        return None, f"ERROR calling Google AI Studio: {e}"

# ---- score badge ----
def _score_badge(score):
    pct = round(score * 100)
    if pct >= 85:
        label = "EXCELLENT"
        color = "#22c55e"
    elif pct >= 65:
        label = "GOOD"
        color = "#f59e0b"
    elif pct >= 40:
        label = "PARTIAL"
        color = "#f97316"
    else:
        label = "INCORRECT"
        color = "#ef4444"
    return (
        '<div style="text-align:center;padding:16px;border-radius:8px;background:'
        + color
        + ';color:#fff;font-size:1.4em;font-weight:700;">'
        + str(pct) + '% - ' + label
        + '</div>'
    )

# ---- main agent loop ----
def run_sentinel(task_id):
    meta   = TASK_META[task_id]
    log    = []
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

    alert = obs.get("alert", "")
    goal  = obs.get("goal", "")
    components = obs.get("components", {})

    log.append(f"\nALERT: {alert}")
    log.append(f"GOAL:  {goal}")
    log.append("\nCOMPONENT STATUS:")
    for name, status in components.items():
        log.append(f"  [{status[:2].upper():>2}] {name}: {status}")
    evidence.append(f"ALERT: {alert}")
    evidence.append(f"COMPONENTS: {json.dumps(components)}")
    yield ("\n".join(log), "", "", "Investigating...")

    # 3. investigation steps
    for action, target in meta["investigate"]:
        time.sleep(0.5)
        result = env_step(action, target)
        if "error" in result:
            log.append(f"\n> {action}({target})\n  ERROR: {result['error']}")
        else:
            obs_text = result.get("observation", str(result))
            log.append(f"\n> {action}({target})")
            for line in obs_text.strip().split("\n"):
                log.append(f"  {line}")
            evidence.append(f"{action}({target}): {obs_text}")
        yield ("\n".join(log), "", "", f"Running {action}({target})...")

    # 4. call Gemma 4
    yield ("\n".join(log), "", "", "Gemma 4 reasoning...")
    evidence_text = "\n".join(evidence)
    parsed, raw = call_gemma(evidence_text, meta["label"])

    log.append("\n--- Gemma 4 raw output ---")
    log.append(raw[:800] if raw else "(empty)")

    if not parsed:
        # smart fallback — pick highest severity component
        severity_order = ["error", "critical", "warn", "degraded"]
        fallback_target = None
        for sev in severity_order:
            for name, status in components.items():
                if sev in status.lower():
                    fallback_target = name
                    break
            if fallback_target:
                break
        if not fallback_target and components:
            fallback_target = list(components.keys())[0]
        parsed = {
            "target": fallback_target or "unknown",
            "root_cause": "Fallback: highest severity component",
            "fix": f"Investigate {fallback_target} logs and metrics",
        }
        log.append(f"WARNING: No valid JSON from Gemma 4. Using fallback target: {fallback_target}")

    # 5. submit diagnosis
    diagnosis_result = env_step("submit_diagnosis", json.dumps(parsed))
    score  = diagnosis_result.get("score", 0.0)
    feedback = diagnosis_result.get("feedback", "")

    log.append(f"\nDIAGNOSIS -> target: {parsed.get('target')}")
    log.append(f"Root cause: {parsed.get('root_cause')}")
    log.append(f"Fix: {parsed.get('fix')}")
    log.append(f"\nSCORE: {score:.4f}")
    log.append(f"Feedback: {feedback}")

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
            gr.Markdown("**Select Incident Scenario**\nEach scenario tests different failure patterns.")
            task_radio = gr.Radio(
                choices=list(TASK_OPTIONS.keys()),
                value="Easy - Data Quality Alert",
                label="",
            )
            task_desc = gr.Textbox(
                value=TASK_META["easy"]["desc"],
                label="",
                interactive=False,
                lines=2,
            )
            run_btn = gr.Button("\U0001F680 Run SentinelAI", variant="primary", size="lg")

        with gr.Column(scale=2):
            status_box = gr.Textbox(value="Select a scenario and click Run SentinelAI.", label="Status", interactive=False)
            with gr.Tabs():
                with gr.TabItem("Investigation Log"):
                    log_box = gr.Textbox(label="Textbox", lines=18, interactive=False)
                with gr.TabItem("Diagnosis JSON"):
                    json_box = gr.Textbox(label="Gemma 4 Diagnosis", lines=10, interactive=False)
                with gr.TabItem("Score"):
                    score_html = gr.HTML()

    task_radio.change(on_task_select, inputs=task_radio, outputs=task_desc)
    run_btn.click(run_from_ui, inputs=task_radio, outputs=[log_box, json_box, score_html, status_box])

    gr.Markdown("""
    ### How it works:
    1. Resets the live RL environment
    2. Investigates: logs, metrics, drift signals
    3. Gemma 4 reads evidence & reasons
    4. Outputs structured JSON diagnosis
    5. Real score returned from environment

    **Model:** `gemma-4-26b-a4b-it` &nbsp; **Env:** [MLOps Incident OpenEnv](https://huggingface.co/spaces/jason9150/mlops-incident-env)
    """)

    gr.DataFrame(
        value=[
            ["Easy - Data Quality",  "Beginner",     "data_pipeline schema change"],
            ["Medium - Latency Spike", "Intermediate", "feature_preprocessor_v2 config"],
            ["Hard - Silent Drift",   "Advanced",     "feature_store PSI spike"],
            ["Cascade Failure",       "Expert",       "deployment v7.8.2 multi-root"],
        ],
        headers=["Scenario", "Difficulty", "Root cause location"],
        label="Scenario Guide",
    )

if __name__ == "__main__":
    demo.launch()
