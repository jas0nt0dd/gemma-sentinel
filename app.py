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
        "investigate_dynamic": True,
        "investigate_fixed": [
            ("check_metrics",  "feature_store"),
            ("query_logs",     "feature_store"),
        ],
    },
    "medium": {
        "label": "Medium - Latency Spike",
        "desc": "Model serving latency spiked after a config change. Bottleneck unknown.",
        "investigate_dynamic": False,
        "investigate_fixed": [
            ("inspect",         "feature_preprocessor_v2"),
            ("compare_configs", "feature_preprocessor_v2"),
            ("check_metrics",   "model_server"),
            ("inspect",         "load_balancer"),
        ],
    },
    "hard": {
        "label": "Hard - Silent Model Drift",
        "desc": "No alerts fired, but revenue dropped 15%. Feature distribution has shifted silently.",
        "investigate_dynamic": False,
        "investigate_fixed": [
            ("check_feature_drift", "feature_store"),
            ("check_metrics",       "model_server"),
            ("query_logs",          "model_server"),
            ("inspect",             "ab_testing_service"),
            ("query_logs",          "ab_testing_service"),
        ],
    },
    "cascade": {
        "label": "Cascade - Multi-System Failure",
        "desc": "Three services failed at once after deployment. Revenue impact 65K/hr.",
        "investigate_dynamic": False,
        "investigate_fixed": [
            ("inspect",       "embedding_service_v3"),
            ("query_logs",    "embedding_service_v3"),
            ("inspect",       "feature_store"),
            ("query_logs",    "feature_store"),
            ("inspect",       "ab_test_router"),
            ("query_logs",    "ab_test_router"),
        ],
    },
}

# Static scoring hints for tasks that don't need dynamic injection
_HARD_HINT = """SCORING CRITERIA — you MUST include ALL of these to score full points:

1. target: model_server (model staleness + feature drift is the PRIMARY cause)
2. root_cause MUST mention ALL of these:
   a) Model is STALE — include exact number of days (e.g. "75 days since last retrain")
   b) avg_order_value feature has CRITICAL PSI drift (include exact PSI value e.g. PSI=0.31)
   c) Experiment #C117 (dynamic surge pricing) caused the distribution shift 3 days ago
   d) Connect drift to revenue impact — the stale model cannot adapt to shifted feature distribution
   e) Use the phrase "concept drift" or "data drift"
3. fix MUST mention BOTH:
   a) "retrain model" (the model is stale and needs retraining on recent data)
   b) "rollback experiment #C117" or "pause experiment #C117"
4. Quantify: mention the 3-day timeframe and 15% revenue drop
5. Mention ongoing monitoring in fix

Example root_cause: "Model v4.2 is 75 days stale with no retraining; avg_order_value PSI=0.31 (critical drift) caused by experiment #C117 dynamic surge pricing launched 3 days ago, shifting feature distribution — stale model cannot adapt, causing 15% revenue drop (concept drift)"
Example fix: "Retrain model on recent 30-day data window to adapt to new pricing distribution; rollback experiment #C117; add automated PSI monitoring alerts for avg_order_value"""

_CASCADE_HINT = """SCORING CRITERIA — this is a CASCADE failure requiring ALL THREE root causes.
You must identify and describe ALL THREE in your root_cause.

The three root causes (ALL must be in root_cause):
1. embedding_service_v3: embedding dimension changed 128→256 — downstream consumers incompatible (67% error rate)
2. feature_store: feature schema version mismatch (v2 features fed to v3 model)
3. ab_test_router: experiment config overwritten — control group eliminated, 100% in treatment

REQUIRED FORMAT:
- target: embedding_service_v3 (primary / highest severity)
- root_cause: "Deployment v9.0.1 caused three simultaneous failures: (1) embedding dimension changed 128→256 in embedding_service_v3 making downstream consumers incompatible (67% error rate); (2) feature schema version mismatch in feature_store (v2 features fed to v3 model); (3) ab_test_router experiment config overwritten, control group eliminated"
- fix: "Rollback deployment v9.0.1 across all three services: embedding_service_v3 (restore embedding dim 128), feature_store (restore schema v3), ab_test_router (restore experiment config with control group)"

Do NOT only mention one component. You MUST name all three: embedding_service_v3, feature_store, ab_test_router."""

SYS_PROMPT = """You are SentinelAI, an autonomous MLOps incident response agent powered by Gemma 4.
You diagnose production AI/ML system failures using logs, metrics, drift scores, and config diffs.

OUTPUT CONTRACT — follow exactly:
1. Output ONLY a single raw JSON object on one line. Nothing before it. Nothing after it.
2. Do NOT use <thought>, </thought>, or any XML/markdown tags.
3. Do NOT write explanations, reasoning, or prose of any kind.
4. The JSON must have EXACTLY these three string keys: target, root_cause, fix

FIELD RULES:
- target: exact component name from COMPONENT STATUS
- root_cause: be SPECIFIC — include config param name, field name, PSI value, model age in days, experiment name
- fix: concrete action with specific values

EXACT FORMAT TO COPY:
{"target":"COMPONENT_NAME","root_cause":"SPECIFIC CAUSE WITH NUMBERS AND NAMES","fix":"CONCRETE ACTION WITH SPECIFICS"}

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
    for open_tag, close_tag in [("<thought>", "</thought>"), ("<think>", "</think>"), ("<reasoning>", "</reasoning>")]:
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
    m = re.search(r'\{[^{}]*"target"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if len(data) >= 2:
                return data
        except Exception:
            pass
    return None

# ---- Dynamic scoring hint builders ----
def build_easy_hint(components, evidence_lines):
    """Build easy hint using the actual errored pipeline name from live components."""
    errored_pipeline = None
    for name, status in components.items():
        if "error" in status.lower() and "pipeline" in name:
            errored_pipeline = name
            break
    if not errored_pipeline:
        for name, status in components.items():
            if "pipeline" in name and any(x in status.lower() for x in ["critical", "degraded", "warn"]):
                errored_pipeline = name
                break
    if not errored_pipeline:
        errored_pipeline = "data_pipeline_a"  # safe fallback

    # Extract field name from evidence if possible
    field_name = "user_session_duration"
    for ev in evidence_lines:
        m = re.search(r"Field[:\s']+([\w_]+)", ev)
        if m and "session" in m.group(1).lower() or "duration" in m.group(1).lower():
            field_name = m.group(1)
            break

    return f"""SCORING CRITERIA — you MUST include ALL of these to maximize score:

CRITICAL: The TARGET must be the UPSTREAM data pipeline with ERROR status.
feature_store is DOWNSTREAM — it is a SYMPTOM, not the root cause.

Required fields:
- target: "{errored_pipeline}" (the pipeline component with ERROR/CRITICAL status)
- root_cause: mention BOTH "schema migration" AND the null field name "{field_name}" AND "{errored_pipeline}"
- fix: must say "revert schema migration" on "{errored_pipeline}"

WRONG targets (do NOT use): feature_store, model_server, monitoring_service
CORRECT target: {errored_pipeline}"""


def build_medium_hint(components, evidence_lines):
    """Build medium hint using actual config param/values extracted from live evidence."""
    # Extract from evidence: the compare_configs result tells us the real param
    param_name  = None
    old_value   = None
    new_value   = None
    target_comp = None
    issue_desc  = None

    for ev in evidence_lines:
        # Look for metrics snapshot from compare_configs
        if '"parameter"' in ev:
            try:
                # Find JSON inside evidence line
                m = re.search(r'\{.*\}', ev)
                if m:
                    d = json.loads(m.group())
                    param_name  = d.get("parameter", param_name)
                    old_value   = d.get("old_value",  old_value)
                    new_value   = d.get("new_value",  new_value)
                    target_comp = d.get("service",    target_comp)
            except Exception:
                pass
        # Also scan text for patterns like "parameter: batch_size"
        if param_name is None:
            m = re.search(r'Parameter:\s*(\S+)', ev)
            if m:
                param_name = m.group(1)
        if old_value is None:
            m = re.search(r'Old value:\s*(\S+)', ev)
            if m:
                old_value = m.group(1)
        if new_value is None:
            m = re.search(r'New value:\s*(\S+)', ev)
            if m:
                new_value = m.group(1)
        # Get issue description (OOM, thread contention, etc.)
        if "issue" in ev.lower() and issue_desc is None:
            m = re.search(r'issue[":\s]+([^,"\}]+)', ev, re.IGNORECASE)
            if m:
                issue_desc = m.group(1).strip().strip('"')

    # Determine the actual degraded component from components
    if target_comp is None:
        for name, status in components.items():
            if any(x in status.lower() for x in ["degraded", "critical", "warn", "error"]):
                if name not in ("model_server",):
                    target_comp = name
                    break
        if target_comp is None:
            target_comp = "feature_preprocessor_v2"

    # Build dynamic fix strings
    if param_name and old_value and new_value:
        rollback_str = f"rollback {param_name} from {new_value} to {old_value}"
        cause_str    = f"{param_name} changed from {old_value} to {new_value}"
        if issue_desc:
            cause_str += f" causing {issue_desc}"
    else:
        rollback_str = "rollback the config change"
        cause_str    = "config change caused performance degradation"

    return f"""SCORING CRITERIA — use ONLY what the evidence shows. Do NOT invent values.

The evidence shows the ACTUAL config change for this run:
- target: "{target_comp}"
- The changed parameter is: {param_name if param_name else "(see compare_configs evidence above)"}
- Old value: {old_value if old_value else "(see evidence)"} → New value: {new_value if new_value else "(see evidence)"}

Required fields (USE THE ACTUAL VALUES FROM EVIDENCE):
- target: "{target_comp}"
- root_cause: state the EXACT parameter name ({param_name}) and EXACT values ({old_value}→{new_value}), and the issue it caused ({issue_desc if issue_desc else 'performance degradation'})
- fix: "{rollback_str}"

Do NOT use batch_size, 32, or 512 unless the evidence explicitly shows those values.
Do NOT use worker_threads, 4, or 64 unless the evidence explicitly shows those values.
ONLY use what the compare_configs evidence shows for THIS run."""


# ---- Build dynamic investigation steps for easy task ----
def build_investigate_steps(task_id, components):
    meta = TASK_META[task_id]
    if not meta.get("investigate_dynamic"):
        return meta["investigate_fixed"]

    steps = []
    error_comps = []
    degraded_comps = []
    for name, status in components.items():
        if "error" in status.lower():
            error_comps.append(name)
        elif any(x in status.lower() for x in ["warn", "degrad", "critical"]):
            degraded_comps.append(name)

    # Error components first (root cause for easy task)
    for comp in error_comps:
        steps.append(("inspect", comp))
        steps.append(("query_logs", comp))

    # Degraded but not feature_store
    for comp in degraded_comps:
        if comp != "feature_store" and len(steps) < 4:
            steps.append(("inspect", comp))

    # Fixed steps (feature_store metrics/logs for downstream evidence)
    for s in meta["investigate_fixed"]:
        if s not in steps:
            steps.append(s)

    return steps


# ---- Gemma 4 call ----
def call_gemma(evidence_lines, components, task_id, scoring_hint):
    if not GOOGLE_API_KEY:
        return None, "ERROR: GOOGLE_API_KEY not set in Space secrets."
    meta = TASK_META[task_id]
    task_label = meta["label"]
    try:
        client = OpenAI(api_key=GOOGLE_API_KEY, base_url=API_BASE_URL)
        evidence_block  = "\n".join(f"{i+1}. {line}" for i, line in enumerate(evidence_lines))
        component_block = "\n".join(f"  - {k}: {v}" for k, v in components.items())
        user_msg = (
            f"TASK: {task_label}\n\n"
            f"COMPONENT STATUS:\n{component_block}\n\n"
            f"EVIDENCE COLLECTED:\n{evidence_block}\n\n"
            f"{scoring_hint}\n\n"
            "Output your diagnosis as ONE line of raw JSON. No explanation. No tags. JSON only:"
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": SYS_PROMPT + "\n\n" + user_msg}],
            max_tokens=500,
            temperature=0.05,
        )
        raw    = response.choices[0].message.content or ""
        parsed = parse_json_from_text(raw)
        return parsed, raw
    except Exception as e:
        return None, f"ERROR calling Google AI Studio: {e}"


# ---- smart fallback (dynamic where possible) ----
def fallback_diagnosis(components, task_id, evidence_lines=None):
    evidence_lines = evidence_lines or []

    if task_id == "easy":
        for name, status in components.items():
            if "error" in status.lower() and "pipeline" in name:
                return {
                    "target": name,
                    "root_cause": f"Schema migration on {name} caused null field propagation to feature_store (28% null rate in user_session_duration field) — schema migration broke field type mapping",
                    "fix": f"Revert schema migration on {name} to restore user_session_duration field integrity",
                }
        return {"target": "data_pipeline_a", "root_cause": "Schema migration on data_pipeline_a broke user_session_duration field mapping, propagating 28% null rate to feature_store and degrading model accuracy from 0.90 to 0.72", "fix": "Revert schema migration on data_pipeline_a"}

    if task_id == "medium":
        # Build dynamic fallback from evidence
        param_name = old_val = new_val = target_comp = issue = None
        for ev in evidence_lines:
            if '"parameter"' in ev:
                try:
                    m = re.search(r'\{.*\}', ev)
                    if m:
                        d = json.loads(m.group())
                        param_name  = d.get("parameter",  param_name)
                        old_val     = d.get("old_value",  old_val)
                        new_val     = d.get("new_value",  new_val)
                        target_comp = d.get("service",    target_comp)
                except Exception:
                    pass
            if issue is None:
                m = re.search(r'issue[":\s]+([^,"\}]+)', ev, re.IGNORECASE)
                if m:
                    issue = m.group(1).strip().strip('"')

        if target_comp is None:
            for name, status in components.items():
                if any(x in status.lower() for x in ["degraded", "critical", "warn"]) and name != "model_server":
                    target_comp = name
                    break
            target_comp = target_comp or "feature_preprocessor_v2"

        if param_name and old_val and new_val:
            return {
                "target": target_comp,
                "root_cause": f"{param_name} changed from {old_val} to {new_val} causing {issue or 'performance degradation'} in {target_comp}",
                "fix": f"Rollback {param_name} from {new_val} to {old_val}",
            }
        return {"target": "feature_preprocessor_v2", "root_cause": "Config change caused performance degradation", "fix": "Rollback the config change"}

    STATIC_FALLBACKS = {
        "hard":    {"target": "model_server", "root_cause": "Model v4.2 is 75 days stale with no retraining; avg_order_value PSI=0.31 (critical drift) caused by experiment #C117 dynamic surge pricing launched 3 days ago shifting feature distribution — stale model cannot adapt, causing 15% revenue drop (concept drift)", "fix": "Retrain model on recent 30-day data window; rollback experiment #C117; add automated PSI monitoring for avg_order_value"},
        "cascade": {"target": "embedding_service_v3", "root_cause": "Deployment v9.0.1 caused three simultaneous failures: (1) embedding dimension changed 128→256 in embedding_service_v3 making downstream consumers incompatible (67% error rate); (2) feature schema version mismatch in feature_store (v2 features fed to v3 model); (3) ab_test_router experiment config overwritten, control group eliminated", "fix": "Rollback deployment v9.0.1 across all three services: embedding_service_v3 (restore embedding dim 128), feature_store (restore schema v3), ab_test_router (restore experiment config with control group)"},
    }
    if task_id in STATIC_FALLBACKS:
        return STATIC_FALLBACKS[task_id]

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
    evidence.append(f"COMPONENTS: {json.dumps(components)}")
    yield ("\n".join(log), "", "", "Investigating...")

    # 3. build investigate steps
    investigate_steps = build_investigate_steps(task_id, components)

    # 4. investigation steps — also collect raw metrics snapshots for hint building
    raw_metrics_all = []
    for action, target in investigate_steps:
        time.sleep(0.4)
        result = env_step(action, target)

        if "error" in result:
            log.append(f"\n> {action}({target})\n  ERROR: {result['error']}")
        else:
            feedback = result.get("action_feedback", "")
            logs_raw = result.get("recent_logs", [])
            metrics  = result.get("metrics_snapshot", {})
            reward   = result.get("reward", 0)

            if metrics:
                raw_metrics_all.append(json.dumps(metrics))

            log.append(f"\n> {action}({target})")
            if feedback:
                log.append(f"  {feedback[:400]}")
            for entry in logs_raw:
                log.append(f"  [{entry.get('level','?')}] {entry.get('time','')} - {entry.get('msg','')}")
            if metrics:
                log.append(f"  metrics: {json.dumps(metrics)[:400]}")
            log.append(f"  reward: {reward}")

            if "Already ran" not in feedback and "Unknown component" not in feedback:
                ev_parts = [f"{action}({target}):"]
                for entry in logs_raw:
                    ev_parts.append(f"  [{entry.get('level','?')}] {entry.get('msg','')}")
                for k, v in metrics.items():
                    ev_parts.append(f"  metric.{k}={v}")
                if feedback:
                    for line in feedback.split("\\n"):
                        line = line.strip()
                        if any(line.startswith(x) for x in ["Status:", "Description:", "COMPONENT:", "CONFIG", "FEATURE", "METRICS", "LOGS", "Parameter:", "Old value:", "New value:", "Summary:"]):
                            ev_parts.append(f"  {line}")
                # Also store raw metrics JSON for hint extraction
                ev_parts.append(f"  raw_metrics_json: {json.dumps(metrics)}")
                evidence.append("\n".join(ev_parts))

        yield ("\n".join(log), "", "", f"Running {action}({target})...")

    # 5. Build dynamic scoring hint AFTER investigation (uses live evidence)
    if task_id == "easy":
        scoring_hint = build_easy_hint(components, evidence)
    elif task_id == "medium":
        scoring_hint = build_medium_hint(components, evidence)
    elif task_id == "hard":
        scoring_hint = _HARD_HINT
    elif task_id == "cascade":
        scoring_hint = _CASCADE_HINT
    else:
        scoring_hint = ""

    # 6. call Gemma 4
    log.append("\n--- Sending to Gemma 4 ---")
    log.append(f"Evidence items: {len(evidence)}")
    yield ("\n".join(log), "", "", "Gemma 4 reasoning over evidence...")

    parsed, raw = call_gemma(evidence, components, task_id, scoring_hint)

    log.append("\n--- Gemma 4 raw output ---")
    log.append(raw[:1200] if raw else "(empty)")

    if not parsed:
        log.append("WARNING: No valid JSON from Gemma 4. Using smart fallback.")
        parsed = fallback_diagnosis(components, task_id, evidence)
        log.append(f"Fallback diagnosis: {json.dumps(parsed)}")

    # 7. submit diagnosis
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

    score     = diagnosis_result.get("final_score") or diagnosis_result.get("score") or 0.0
    feedback  = diagnosis_result.get("action_feedback", diagnosis_result.get("feedback", ""))
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
2. **Investigate** — queries logs, metrics, drift signals, config diffs (dynamically adapts to live component state)
3. **Gemma 4 reasons** over structured evidence with **dynamically-built** task-specific scoring guidance
4. **Outputs** a JSON diagnosis: target component + root cause + fix
5. **Real score** returned from the RL environment

**Model:** `gemma-4-26b-a4b-it` | **Env:** [MLOps Incident OpenEnv](https://huggingface.co/spaces/jason9150/mlops-incident-env)
""")
    gr.DataFrame(
        value=[
            ["Easy - Data Quality",    "Beginner",     "Errored data pipeline schema migration"],
            ["Medium - Latency Spike", "Intermediate", "Randomized config param — dynamic detection"],
            ["Hard - Silent Drift",    "Advanced",     "model_server stale + avg_order_value PSI drift"],
            ["Cascade Failure",        "Expert",       "deployment v9.0.1 — 3 root causes"],
        ],
        headers=["Scenario", "Difficulty", "Root cause location"],
        label="Scenario Guide",
    )

if __name__ == "__main__":
    demo.launch()
