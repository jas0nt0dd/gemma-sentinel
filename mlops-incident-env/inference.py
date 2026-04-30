"""
inference.py - MLOps Incident Response | LLM-driven agent

The agent reads observations, keeps conversation memory, and decides every action.
There are no task-specific scripted paths or hidden keyword-injected diagnoses.

Oracle distillation: set ORACLE_TRACE_PATH=traces/oracle.jsonl (optional ORACLE_TRACE_MIN_SCORE=0.9)
to append successful episode traces. Convert with scripts/oracle_traces_to_sft_jsonl.py then
SFT_ORACLE_JSONL=... before GRPO in hf_train.py (SFT first, then RL).
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import requests
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:8000").rstrip("/")

# Append gold-standard oracle trajectories (full message history) for distillation → SFT → GRPO.
ORACLE_TRACE_PATH = os.getenv("ORACLE_TRACE_PATH", "").strip()
ORACLE_TRACE_MIN_SCORE = float(os.getenv("ORACLE_TRACE_MIN_SCORE", "0.9"))

ENV_NAME = "mlops-incident-env"
MAX_STEPS = 12
TEMPERATURE = 0.2
MAX_TOKENS = 700
PROMPT_JSON_CHARS = 1800
PROMPT_HISTORY_LINES = 8
HISTORY_FEEDBACK_CHARS = 650
TASKS = ["easy", "medium", "hard", "cascade"]
TASK_STEP_BUDGET = {
    "easy": 10,
    "medium": 15,
    "hard": 25,
    "cascade": 30,
}
TASK_FORCE_SUBMIT_AT = {
    "easy": 4,
    "medium": 5,
    "hard": 6,
    "cascade": 8,
}
HTTP_RETRY_ATTEMPTS = 3
HTTP_TIMEOUT_SECONDS = 45
LLM_RETRY_ATTEMPTS = 3
BLOCKED_ACTIONS = {"request_rollback"}
DRIFT_ALLOWED_COMPONENT_HINTS = (
    "feature_store",
    "model_server",
    "model_serving",
    "ab_testing_service",
    "training_pipeline",
)
LATENCY_COMPONENT_HINTS = (
    "preprocessor",
    "model_serving",
    "model_server",
)

VALID_ACTIONS = [
    "inspect",
    "query_logs",
    "check_metrics",
    "compare_configs",
    "check_feature_drift",
    "submit_diagnosis",
    "request_rollback",
]
INVESTIGATION_ACTIONS = ["inspect", "query_logs", "check_metrics", "compare_configs"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous MLOps incident response agent.

    Your job is to investigate production ML incidents using only observations returned
    by the environment, then submit a diagnosis when there is enough evidence or when
    the step budget is nearly exhausted.

    Investigation strategy:
    - Start with components in degraded, error, warning, or critical states.
    - Use inspect to understand a component, query_logs for concrete error evidence,
      check_metrics for quantitative symptoms, and compare_configs when a deployment
      or configuration change is likely.
    - Use check_feature_drift for silent degradation, revenue drops, model quality
      drops without infrastructure errors, PSI, feature distribution shift, or stale
      model symptoms.
    - For multi-system incidents, gather evidence from every degraded subsystem before
      diagnosing, then connect them to the shared deployment or upstream change.
    - Never repeat the same action on the same component.

    Diagnosis quality:
    - Use exact component names and exact evidence values found in observations.
    - For data quality incidents, name the broken pipeline, schema/field issue, and
      downstream impact.
    - For latency/config incidents, name the bottleneck service, changed parameter,
      old/new values if observed, symptom, and rollback or config fix.
    - For drift incidents, name the drifted feature, PSI/distribution evidence, related
      experiment or product change if observed, model staleness, business impact, and
      retraining/monitoring plan.
    - For cascade incidents, name all root-cause services, the issue in each one, the
      common deployment or rollout, and a coordinated rollback plan.

    Output exactly one JSON object and no prose:
    {
      "action_type": "inspect|query_logs|check_metrics|compare_configs|check_feature_drift|submit_diagnosis|request_rollback",
      "target": "exact_component_name_or_diagnosis_target",
      "parameters": {}
    }

    For submit_diagnosis, parameters must be:
    {
      "root_cause": "your evidence-based root cause",
      "fix": "your remediation plan"
    }
    """
).strip()


# Data model
@dataclass
class Obs:
    goal: str = ""
    alert_summary: str = ""
    component_status: Dict[str, str] = field(default_factory=dict)
    recent_logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    action_feedback: str = ""
    step_count: int = 0
    sla_minutes_remaining: Optional[int] = None
    reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    final_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None


def _to_obs(d: dict) -> Obs:
    return Obs(
        goal=d.get("goal", ""),
        alert_summary=d.get("alert_summary", ""),
        component_status=d.get("component_status", {}),
        recent_logs=d.get("recent_logs", []),
        metrics_snapshot=d.get("metrics_snapshot", {}),
        action_feedback=d.get("action_feedback", ""),
        step_count=int(d.get("step_count", 0)),
        sla_minutes_remaining=d.get("sla_minutes_remaining"),
        reward=float(d.get("reward", 0.0)),
        cumulative_reward=float(d.get("cumulative_reward", 0.0)),
        done=bool(d.get("done", False)),
        final_score=d.get("final_score"),
        score_breakdown=d.get("score_breakdown"),
    )


# HTTP client
class DirectEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post_with_retry(self, path: str, payload: dict) -> Obs:
        last_error: Optional[Exception] = None
        request_payload = dict(payload)
        request_payload.setdefault("request_id", uuid.uuid4().hex)
        for attempt in range(HTTP_RETRY_ATTEMPTS):
            try:
                r = requests.post(
                    f"{self.base_url}/{path.lstrip('/')}",
                    json=request_payload,
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                r.raise_for_status()
                return _to_obs(r.json())
            except Exception as e:
                last_error = e
                if attempt == HTTP_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(3 + attempt * 2)

        raise RuntimeError("Request retry loop exited unexpectedly") from last_error

    def reset(self, task_id: str = "easy") -> Obs:
        return self._post_with_retry("reset", {"task_id": task_id})

    def step(self, action_type: str, target: str, parameters: dict) -> Obs:
        return self._post_with_retry(
            "step",
            {"action_type": action_type, "target": target, "parameters": parameters},
        )

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=10).status_code == 200
        except Exception:
            return False


def resolve_tasks() -> List[str]:
    requested = os.getenv("TASKS", "").strip()
    if not requested:
        return list(TASKS)

    valid_tasks = set(TASKS)
    selected = [task.strip().lower() for task in requested.split(",") if task.strip()]
    filtered = [task for task in selected if task in valid_tasks]
    return filtered or list(TASKS)


def create_completion_with_retry(
    llm: OpenAI,
    messages: List[ChatCompletionMessageParam],
):
    last_error: Optional[Exception] = None
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            return llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_for_model(messages),
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
        except Exception as e:
            last_error = e
            error_text = _norm(str(e))
            is_retryable = any(
                term in error_text
                for term in ["429", "rate limit", "timeout", "timed out", "connection", "temporary"]
            )
            if attempt == LLM_RETRY_ATTEMPTS - 1 or not is_retryable:
                raise
            time.sleep(5 + attempt * 5)

    raise RuntimeError("LLM retry loop exited unexpectedly") from last_error


# Prompting
def build_user_prompt(
    obs: Obs,
    history: List[str],
    seen: set[str],
    remaining_steps: int,
    task_id: str,
) -> str:
    status = "\n".join(f"  - {name}: {state}" for name, state in obs.component_status.items())
    logs = _json_block(obs.recent_logs) if obs.recent_logs else "None returned by the last action."
    metrics = _json_block(obs.metrics_snapshot) if obs.metrics_snapshot else "None returned by the last action."
    history_text = "\n".join(history[-PROMPT_HISTORY_LINES:]) if history else "No actions taken yet."
    seen_text = ", ".join(sorted(seen)) if seen else "None"
    readiness = diagnosis_readiness(obs, history, seen, remaining_steps)
    force_submit_at = TASK_FORCE_SUBMIT_AT.get(task_id, 3)
    suggested_action, suggested_target, _ = fallback_action(
        obs, list(obs.component_status.keys()), seen, history, remaining_steps, task_id
    )

    urgency = ""
    if remaining_steps <= force_submit_at:
        urgency = (
            f"\nURGENCY: only {remaining_steps} step(s) remain. If you have enough evidence, "
            f"submit_diagnosis now. Forced fallback submission begins at remaining_steps <= {force_submit_at}."
        )

    return textwrap.dedent(
        f"""
        INCIDENT GOAL:
        {obs.goal}

        ALERT:
        {obs.alert_summary}

        STEP:
        current_step={obs.step_count}
        remaining_steps={remaining_steps}
        sla_minutes_remaining={obs.sla_minutes_remaining}
        reward={obs.reward}
        cumulative_reward={obs.cumulative_reward}
        {urgency}

        COMPONENT STATUS:
        {status}

        LATEST ACTION FEEDBACK:
        {compact_text(obs.action_feedback or "None", 1400)}

        RECENT LOGS:
        {logs}

        METRICS SNAPSHOT:
        {metrics}

        INVESTIGATION HISTORY:
        {history_text}

        USED ACTION:COMPONENT PAIRS:
        {seen_text}

        DIAGNOSIS READINESS:
        {readiness}

        SUGGESTED INVESTIGATION FOCUS:
        If you are not ready to diagnose, prefer an unseen evidence-gathering action like
        {suggested_action} on {suggested_target}.

        AVAILABLE ACTIONS:
        {", ".join(VALID_ACTIONS)}

        Return exactly one JSON object for the next action. Do not repeat a used
        action:component pair. Use exact component names from COMPONENT STATUS when
        targeting a component.
        """
    ).strip()


def _json_block(value: Any) -> str:
    return compact_text(json.dumps(value, sort_keys=True), PROMPT_JSON_CHARS)


def compact_text(value: str, limit: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[:limit] + " ...[truncated]"


# Action parsing and normalization
def parse_action(text: str, valid_components: Sequence[str]) -> Optional[Tuple[str, str, dict]]:
    obj = _extract_json_object(text)
    if not obj:
        return None

    action_type = normalize_action(obj.get("action_type", ""))
    if action_type not in VALID_ACTIONS:
        return None

    target = str(obj.get("target", "") or "")
    parameters = obj.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    if action_type == "submit_diagnosis":
        if not parameters.get("root_cause"):
            return None
        if not parameters.get("fix"):
            parameters["fix"] = "Mitigate the identified root cause and monitor recovery."
        if target and valid_components:
            target = _fuzzy_match(target, valid_components)
        elif valid_components:
            target = valid_components[0]
        return action_type, target, parameters

    if not valid_components:
        return action_type, target, parameters

    if action_type == "check_metrics" and _norm(target) in {"global", "business", "revenue"}:
        target = _norm(target)
    elif action_type == "check_feature_drift" and not target:
        target = next((c for c in valid_components if "feature" in _norm(c)), valid_components[0])
    else:
        target = _fuzzy_match(target, valid_components)

    return action_type, target, parameters


def _extract_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    loaded = json.loads(text[start : i + 1])
                    return loaded if isinstance(loaded, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def normalize_action(action_type: str) -> str:
    compact = _norm(action_type).replace(" ", "")
    aliases = {
        "inspect": "inspect",
        "querylogs": "query_logs",
        "querylog": "query_logs",
        "checklogs": "query_logs",
        "checkmetrics": "check_metrics",
        "metrics": "check_metrics",
        "compareconfigs": "compare_configs",
        "compareconfig": "compare_configs",
        "configdiff": "compare_configs",
        "checkfeaturedrift": "check_feature_drift",
        "featuredrift": "check_feature_drift",
        "submitdiagnosis": "submit_diagnosis",
        "diagnose": "submit_diagnosis",
        "requestrollback": "request_rollback",
        "rollback": "request_rollback",
    }
    return aliases.get(compact, action_type.lower().replace("-", "_").replace(" ", "_"))


def _fuzzy_match(name: str, valid: Sequence[str]) -> str:
    """Resolve approximate component names such as feature_preprocessor -> exact component."""
    if not valid:
        return name
    if name in valid:
        return name

    target_norm = _norm(name)
    target_compact = target_norm.replace(" ", "")
    if not target_compact:
        return valid[0]

    for component in valid:
        if _norm(component) == target_norm:
            return component

    for component in valid:
        comp_compact = _norm(component).replace(" ", "")
        if target_compact in comp_compact or comp_compact in target_compact:
            return component

    target_tokens = [t for t in target_norm.split() if len(t) > 2]
    best = valid[0]
    best_score = -1.0
    for component in valid:
        comp_tokens = set(_norm(component).split())
        overlap = sum(1 for token in target_tokens if token in comp_tokens)
        score = overlap / max(len(target_tokens), 1)
        if any(token in _norm(component) for token in target_tokens):
            score += 0.25
        if score > best_score:
            best_score = score
            best = component

    return best


def _norm(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def incident_kind(obs: Obs) -> str:
    text = _norm(f"{obs.goal} {obs.alert_summary}")
    if any(term in text for term in ["three ml services", "multi system", "cascade", "all root causes"]):
        return "cascade"
    if any(term in text for term in ["revenue", "silent", "drift", "distribution", "no error alerts"]):
        return "drift"
    if any(term in text for term in ["latency", "timeout", "p99", "bottleneck", "config change"]):
        return "latency"
    if any(term in text for term in ["accuracy", "data quality", "schema", "null", "pipeline"]):
        return "data_quality"
    return "general"


def diagnosis_readiness(
    obs: Obs,
    history: List[str],
    seen: set[str],
    remaining_steps: int,
) -> str:
    if remaining_steps <= 2:
        return "URGENT: diagnose soon, but include only evidence actually observed."

    kind = incident_kind(obs)
    evidence = evidence_text(obs, history)
    inspected_or_logged = [key for key in seen if key.startswith(("inspect:", "query_logs:", "check_metrics:", "compare_configs:", "check_feature_drift:"))]

    if kind == "data_quality":
        has_pipeline_logs = any(key.startswith("query_logs:") and "pipeline" in key for key in seen)
        has_schema_evidence = any(term in evidence for term in ["schema", "validation failed", "null", "field", "migration"])
        has_downstream = any("feature_store" in key or "model_server" in key for key in seen)
        if has_pipeline_logs and has_schema_evidence:
            if has_downstream:
                return "READY: pipeline/schema evidence and downstream impact evidence have been observed."
            return "READY: pipeline/schema evidence sufficient. Optionally verify feature_store impact then submit."
        return (
            "NOT READY: before submit_diagnosis, identify the broken data pipeline from logs, "
            "capture the schema/field/null evidence, and optionally verify downstream impact in feature_store or model_server."
        )

    if kind == "latency":
        has_config = any(key.startswith("compare_configs:") for key in seen)
        has_config_detail = any(
            term in evidence
            for term in [
                "parameter",
                "old value",
                "new value",
                "batch size",
                "worker threads",
                "max concurrent requests",
            ]
        )
        has_service_evidence = any(term in evidence for term in ["deployed", "latency", "timeout"])
        if has_config and has_config_detail and has_service_evidence and len(inspected_or_logged) >= 3:
            return "READY: config diff plus latency/service evidence have been observed."
        return (
            "NOT READY: compare configs and collect logs/metrics on the slow service. "
            "The final diagnosis must name the exact changed parameter and old/new value if observed."
        )

    if kind == "drift":
        has_drift = any(key.startswith("check_feature_drift:") for key in seen)
        has_critical_feature = any(term in evidence for term in ["critical drift", "critical_drift", "psi"])
        has_model_age = any(term in evidence for term in ["trained", "retrained", "model age", "days ago", "stale"])
        has_experiment = any(term in evidence for term in ["experiment", "redesign", "pricing", "checkout", "ab test"])
        if has_drift and has_critical_feature and has_model_age and has_experiment:
            return "READY: drift feature/PSI, model age, and experiment or product-change evidence have been observed."
        return (
            "NOT READY: cite the CRITICAL_DRIFT feature and PSI, then collect model staleness "
            "and experiment/product-change evidence before submit_diagnosis."
        )

    if kind == "cascade":
        bad_components = {
            component for component, status in obs.component_status.items()
            if any(term in _norm(status) for term in ["critical", "error", "degraded", "failed"])
            and "model_serving" not in component
            and "model_server" not in component
        }
        covered = {
            key.split(":", 1)[1] for key in seen
            if key.startswith(("query_logs:", "check_metrics:"))
        } & bad_components
        if len(covered) >= min(3, len(bad_components)):
            return "READY: logs or metrics were gathered from all primary degraded root-cause subsystems."
        missing = sorted(bad_components - covered)
        return (
            "NOT READY: gather logs or metrics from every primary degraded subsystem "
            f"before diagnosis. Missing evidence from: {', '.join(missing) or 'unknown'}."
        )

    if len(inspected_or_logged) >= 3:
        return "READY: multiple evidence-gathering actions have been taken."
    return "NOT READY: gather logs, metrics, or config evidence before submit_diagnosis."


def submit_ready(obs: Obs, history: List[str], seen: set[str], remaining_steps: int) -> bool:
    if remaining_steps <= 2:
        return True
    return diagnosis_readiness(obs, history, seen, remaining_steps).startswith("READY")


def should_force_submit(task_id: str, remaining_steps: int) -> bool:
    return remaining_steps <= TASK_FORCE_SUBMIT_AT.get(task_id, 3)


def diagnosis_plausible(
    action_type: str,
    target: str,
    parameters: dict,
    obs: Obs,
    history: List[str],
) -> bool:
    if action_type != "submit_diagnosis":
        return True

    text = _norm(f"{target} {parameters}")
    evidence = evidence_text(obs, history)
    kind = incident_kind(obs)

    if kind == "data_quality":
        expected_target = _infer_data_quality_target(obs, list(obs.component_status.keys()), history)
        target_valid = not expected_target or _norm(target) == _norm(expected_target)
        return target_valid and "pipeline" in text and any(
            term in text for term in ["schema", "null", "field", "migration", "validation"]
        )
    if kind == "latency":
        expected_target = _infer_latency_target(obs, list(obs.component_status.keys()), history)
        target_valid = not expected_target or _norm(target) == _norm(expected_target)
        names_changed_parameter = any(
            term in text
            for term in [
                "batch size",
                "worker threads",
                "max concurrent requests",
            ]
        )
        names_changed_value = bool(re.search(r"\b\d+\b", text))
        names_remediation = any(term in text for term in ["rollback", "revert", "restore", "previous config"])
        ties_to_latency = any(term in text for term in ["latency", "timeout", "queue", "oom", "memory", "cpu", "contention"])
        signal_count = sum([names_changed_parameter, names_changed_value, names_remediation, ties_to_latency])
        return target_valid and signal_count >= 2
    if kind == "drift":
        return "drift" in text and "psi" in text and any(term in text for term in ["retrain", "stale", "trained", "experiment", "redesign", "pricing", "checkout"])
    if kind == "cascade":
        bad_components = [
            component for component, status in obs.component_status.items()
            if any(term in _norm(status) for term in ["critical", "error", "degraded"])
            and "model_serving" not in component
            and "model_server" not in component
        ]
        mentioned = sum(1 for component in bad_components if _norm(component) in text)
        return mentioned >= min(3, len(bad_components)) and any(term in text for term in ["rollback", "deployment", "deploy", "coordinated"])

    return len(text) > 80


def evidence_text(obs: Obs, history: List[str]) -> str:
    return _norm(raw_evidence_text(obs, history))


def raw_evidence_text(obs: Obs, history: List[str]) -> str:
    return (
        f"{obs.goal} {obs.alert_summary} "
        + " ".join(history)
        + " "
        + (obs.action_feedback or "")
    )


def is_allowed_drift_component(target: str) -> bool:
    target_text = _norm(target).replace(" ", "_")
    return any(hint in target_text for hint in DRIFT_ALLOWED_COMPONENT_HINTS)


def _action_targets(history: List[str], action_name: str) -> List[str]:
    pattern = re.compile(rf"{re.escape(action_name)}\(([^)]+)\)")
    targets: List[str] = []
    for entry in history:
        match = pattern.search(entry)
        if match:
            targets.append(match.group(1))
    return targets


def _status_priority(status: str) -> int:
    status_text = _norm(status)
    if any(term in status_text for term in ["critical", "error", "failed"]):
        return 0
    if any(term in status_text for term in ["degraded", "warning", "warn"]):
        return 1
    if "healthy" in status_text or "ok" in status_text:
        return 3
    return 2


def _infer_data_quality_target(
    obs: Obs,
    valid_components: Sequence[str],
    history: List[str],
) -> str:
    pipeline_components = [c for c in valid_components if "pipeline" in _norm(c)]
    if not pipeline_components:
        return ""

    investigated = set()
    for action_name in ("query_logs", "inspect", "check_metrics"):
        investigated.update(target for target in _action_targets(history, action_name) if target in pipeline_components)

    return sorted(
        pipeline_components,
        key=lambda component: (
            _status_priority(obs.component_status.get(component, "")),
            0 if component in investigated else 1,
            component,
        ),
    )[0]


def _infer_latency_target(
    obs: Obs,
    valid_components: Sequence[str],
    history: List[str],
) -> str:
    compare_targets = [
        target for target in _action_targets(history, "compare_configs")
        if target in valid_components
    ]
    ranked_targets = [
        target for target in compare_targets
        if any(hint in _norm(target).replace(" ", "_") for hint in LATENCY_COMPONENT_HINTS)
    ]
    if ranked_targets:
        return sorted(
            ranked_targets,
            key=lambda component: (
                _status_priority(obs.component_status.get(component, "")),
                compare_targets.index(component),
            ),
        )[0]

    latency_components = [
        component for component in valid_components
        if any(hint in _norm(component).replace(" ", "_") for hint in LATENCY_COMPONENT_HINTS)
    ]
    if not latency_components:
        return ""

    return sorted(
        latency_components,
        key=lambda component: (
            _status_priority(obs.component_status.get(component, "")),
            component,
        ),
    )[0]


# Fallback and loop guard
def fallback_action(
    obs: Obs,
    valid_components: Sequence[str],
    seen: set[str],
    history: List[str],
    remaining_steps: int,
    task_id: str,
) -> Tuple[str, str, dict]:
    """Choose a genuinely unseen action when the LLM response is invalid or loops."""
    if remaining_steps <= 1:
        return fallback_diagnosis(obs, valid_components, history)

    if should_force_submit(task_id, remaining_steps):
        return fallback_diagnosis(obs, valid_components, history)

    candidates = fallback_candidates(obs, valid_components)
    for action_type, target in candidates:
        if f"{action_type}:{target}" not in seen:
            return action_type, target, {}

    return fallback_diagnosis(obs, valid_components, history)


def fallback_investigation_action(
    obs: Obs,
    valid_components: Sequence[str],
    seen: set[str],
) -> Tuple[str, str, dict]:
    for action_type, target in fallback_candidates(obs, valid_components):
        if action_type in INVESTIGATION_ACTIONS and f"{action_type}:{target}" not in seen:
            return action_type, target, {}

    target = prioritized_components(obs, valid_components)[0] if valid_components else ""
    return "inspect", target, {}


def fallback_diagnosis(
    obs: Obs,
    valid_components: Sequence[str],
    history: List[str],
) -> Tuple[str, str, dict]:
    kind = incident_kind(obs)
    evidence = evidence_summary(obs, history)
    raw_evidence = raw_evidence_text(obs, history)
    normalized_evidence = evidence_text(obs, history)
    target = best_evidence_target(obs, valid_components, history)

    if kind == "data_quality":
        target = _infer_data_quality_target(obs, valid_components, history) or target
        field = _extract_data_quality_field(raw_evidence) or "the affected transaction feature"
        migration = "schema migration" if "migration" in normalized_evidence else "schema change"
        evidence = (
            f"{target} broke after a {migration}: {field} no longer matched the expected schema, "
            "causing validation failures, dropped or null values in feature_store, and degraded model accuracy."
        )
        fix = "Rollback or revert the bad schema migration, backfill feature_store data, and monitor model accuracy recovery."
    elif kind == "latency":
        target = _infer_latency_target(obs, valid_components, history) or target
        parameter = _extract_latency_parameter(raw_evidence)
        new_value = _extract_latency_new_value(raw_evidence, parameter)
        old_value = _extract_latency_old_value(raw_evidence, parameter) or _extract_value(
            raw_evidence, rf"{re.escape(parameter)}\s*(\d+)\s*[-=]>\s*(\d+)", group=1
        )
        symptom = _extract_latency_symptom(raw_evidence)
        value_text = f"={new_value}" if new_value else ""
        rollback_text = (
            f" Restore the previous {parameter}={old_value}."
            if old_value else
            f" Roll back to the previous stable {parameter}."
        )
        evidence = (
            f"{target} was deployed with {parameter}{value_text}, which drove {symptom} and request timeouts. "
            "This is the latency bottleneck, not the load_balancer."
        )
        fix = (
            f"Rollback the risky {parameter} change on {target}, reduce {symptom}, and monitor latency and timeout recovery."
            + rollback_text
        )
    elif kind == "drift":
        feature = _extract_feature_name(raw_evidence, ["user_engagement_score"]) or "user_engagement_score"
        target = (
            _find_component(valid_components, "ab_testing_service", "ab testing service")
            or _find_component(valid_components, "model_server", "model serving")
            or _find_component(valid_components, "training_pipeline", "training pipeline")
            or best_evidence_target(obs, valid_components, history)
        )
        psi_value = _extract_psi(raw_evidence)
        experiment = _extract_experiment_name(raw_evidence)
        model_age = _extract_model_age(raw_evidence)
        psi_text = f"PSI={psi_value}" if psi_value else "PSI above threshold"
        experiment_text = experiment or "ui_redesign"
        stale_text = f"stale for {model_age}" if model_age else "stale"
        evidence = (
            f"{feature} drifted ({psi_text}) after {experiment_text}; the model is {stale_text} and was not retrained for the new distribution. "
            "That drift is driving the silent revenue decline."
        )
        fix = (
            "Retrain with a recent post-change data window, validate business metrics after retraining, and enable continuous PSI drift monitoring."
        )
    elif kind == "cascade":
        target = _find_component(valid_components, "embedding_service_v3", "embedding service") or target
        deployment = _extract_deployment_name(raw_evidence) or "deployment"
        causes = _extract_cascade_causes(valid_components, raw_evidence)
        evidence = (
            f"{deployment} introduced a cascade failure: {', '.join(causes)}. "
            "All three failures must be addressed together."
        )
        fix = "Perform a coordinated rollback of all affected services to the pre-deployment state, then verify each subsystem and model quality."
    else:
        fix = "Apply the safest remediation for the identified failing component, rollback recent risky changes if present, and monitor recovery."

    target = validate_component_target(target, valid_components, obs, history)
    return "submit_diagnosis", target, {"root_cause": evidence, "fix": fix}


def _find_component(valid_components: Sequence[str], *terms: str) -> str:
    normalized_terms = [_norm(term) for term in terms]
    for component in valid_components:
        component_text = _norm(component)
        if any(term and term in component_text for term in normalized_terms):
            return component
    return ""


def validate_component_target(
    target: str,
    valid_components: Sequence[str],
    obs: Obs,
    history: List[str],
) -> str:
    if not valid_components:
        return target
    if target in valid_components:
        return target

    matched = _find_component(valid_components, target)
    if matched:
        return matched

    return best_evidence_target(obs, valid_components, history)


def _extract_value(text: str, pattern: str, group: int = 1) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(group)


def _extract_data_quality_field(text: str) -> str:
    patterns = [
        r"field\s*['\"]?([a-zA-Z0-9_]+)['\"]?",
        r"for\s+['\"]?([a-zA-Z0-9_]+)['\"]?",
        r"missing\s+([a-zA-Z0-9_]+)",
        r"corrupted\s+([a-zA-Z0-9_]+)",
    ]
    for pattern in patterns:
        value = _extract_value(text, pattern)
        if value:
            return value
    return ""


def _extract_latency_parameter(text: str) -> str:
    parameter = _extract_value(text, r"parameter\s*[:=]?\s*([a-zA-Z0-9_]+)")
    if parameter:
        return parameter

    normalized = _norm(text)
    known_parameters = (
        "batch_size",
        "worker_threads",
        "max_concurrent_requests",
    )
    for parameter_name in known_parameters:
        if _norm(parameter_name) in normalized:
            return parameter_name
    return "configuration"


def _extract_latency_new_value(text: str, parameter: str) -> str:
    return (
        _extract_value(text, r"new value\s*[:=]?\s*([a-zA-Z0-9_.-]+)")
        or _extract_value(text, rf"{re.escape(parameter)}\s*[=:]?\s*([a-zA-Z0-9_.-]+)")
        or _extract_value(text, rf"{re.escape(parameter)}\s*[:=]?\s*([0-9]+)\s*(?:->|→)\s*([0-9]+)", group=2)
    )


def _extract_latency_old_value(text: str, parameter: str) -> str:
    return (
        _extract_value(text, r"old value\s*[:=]?\s*([a-zA-Z0-9_.-]+)")
        or _extract_value(text, rf"previous\s+{re.escape(parameter)}\s*[=:]?\s*([a-zA-Z0-9_.-]+)")
        or _extract_value(text, rf"{re.escape(parameter)}\s*[:=]?\s*([0-9]+)\s*(?:->|→)\s*([0-9]+)", group=1)
    )


def _extract_latency_symptom(text: str) -> str:
    normalized = _norm(text)
    if any(term in normalized for term in ["oom", "memory leak", "memory pressure", "memory"]):
        return "memory pressure and OOM risk"
    if any(term in normalized for term in ["thread contention", "cpu saturation", "cpu"]):
        return "thread contention and CPU saturation"
    if any(term in normalized for term in ["request queue", "queue overflow", "concurrent"]):
        return "request queue overflow"
    return "latency spikes"


def _extract_psi(text: str) -> str:
    return _extract_value(text, r"\bpsi\b\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)")


def _extract_model_age(text: str) -> str:
    prioritized_patterns = [
        r"(?:last retrained|last retrain|model (?:was )?trained|model age)[^0-9]{0,24}(\d+)\s+days?\b",
        r"(?:trained|retrained)[^0-9]{0,24}(\d+)\s+days?\b",
    ]
    for pattern in prioritized_patterns:
        days = _extract_value(text, pattern)
        if days:
            return f"{days} days"

    matches = re.findall(r"\b(\d+)\s+days?\b", text, flags=re.IGNORECASE)
    if not matches:
        return ""
    return f"{max(int(day) for day in matches)} days"


def _extract_experiment_name(text: str) -> str:
    experiment = _extract_value(text, r"\b([A-Z]\d{2,})\b")
    if experiment:
        return experiment
    if re.search(r"\bui redesign\b", text, flags=re.IGNORECASE):
        return "ui_redesign"
    return ""


def _extract_feature_name(text: str, defaults: Sequence[str]) -> str:
    critical_match = re.search(
        r"\b([a-z][a-z0-9_]+)\s*:\s*psi\s*(?:=|:)\s*[0-9]+(?:\.[0-9]+)?\s*\[critical[\s_]?drift\]",
        text,
        flags=re.IGNORECASE,
    )
    if critical_match:
        return critical_match.group(1)

    drifted_match = re.search(
        r"\b([a-z][a-z0-9_]{4,})\s+drifted\b",
        text,
        flags=re.IGNORECASE,
    )
    if drifted_match:
        return drifted_match.group(1)

    normalized_text = _norm(text)
    for feature in defaults:
        if _norm(feature) in normalized_text:
            return feature

    feature_match = re.search(
        r"\b([a-z][a-z0-9_]+)\s*:\s*psi\s*(?:=|:)\s*[0-9]+(?:\.[0-9]+)?",
        text,
        flags=re.IGNORECASE,
    )
    if feature_match:
        return feature_match.group(1)
    return ""


def _extract_deployment_name(text: str) -> str:
    deployment = _extract_value(text, r"\b(deploy(?:ment)?\s*v?[0-9]+(?:\.[0-9]+){1,3})\b")
    if deployment:
        return deployment
    if re.search(r"\bdeployment\b", text, flags=re.IGNORECASE):
        return "deployment"
    return ""


def _extract_cascade_causes(valid_components: Sequence[str], text: str) -> List[str]:
    embedding = _find_component(valid_components, "embedding_service_v3", "embedding service") or "embedding_service_v3"
    feature_store = _find_component(valid_components, "feature_store", "feature store") or "feature_store"
    primary_services = [
        component for component in valid_components
        if component not in {embedding, feature_store}
        and "model_serving" not in component
        and "model_server" not in component
    ]
    third_service = primary_services[0] if primary_services else (
        _find_component(valid_components, "ab_test_router", "ab test router")
        or "ab_test_router"
    )

    embedding_issue = (
        f"{embedding} had an ONNX runtime mismatch that corrupted embeddings"
        if re.search(r"\bonnx\b", text, flags=re.IGNORECASE)
        else (
            f"{embedding} lost GPU compatibility and fell back to CPU"
            if re.search(r"\bcuda\b|\bgpu unavailable\b", text, flags=re.IGNORECASE)
            else (
                f"{embedding} changed embedding dimensions and broke downstream consumers"
                if re.search(r"\bdimension\b", text, flags=re.IGNORECASE)
                else f"{embedding} corrupted embeddings after the deployment"
            )
        )
    )
    feature_store_issue = (
        f"{feature_store} set cache TTL=0 and served stale features"
        if re.search(r"\bttl\b|\bcache\b", text, flags=re.IGNORECASE)
        else (
            f"{feature_store} serialized feature reads through an undersized Redis connection pool"
            if re.search(r"\bredis\b|\bconnection pool\b|\bserialized\b", text, flags=re.IGNORECASE)
            else (
                f"{feature_store} sent the wrong feature schema version to the model"
                if re.search(r"\bschema version mismatch\b", text, flags=re.IGNORECASE)
                else f"{feature_store} served stale features after the deployment"
            )
        )
    )
    third_issue = (
        f"{third_service} sent 100% of traffic to untested model_B"
        if re.search(r"\b100%\b|\btraffic split\b|\bmodel[_ ]b\b", text, flags=re.IGNORECASE)
        else (
            f"{third_service} pointed production at the wrong staging model artifact"
            if re.search(r"\bartifact\b|\bstaging model\b|\bwrong model\b", text, flags=re.IGNORECASE)
            else (
                f"{third_service} overwrote the experiment config and eliminated the control group"
                if re.search(r"\bexperiment config\b|\bcontrol group\b", text, flags=re.IGNORECASE)
                else f"{third_service} corrupted the routing split"
            )
        )
    )
    return [embedding_issue, feature_store_issue, third_issue]


def best_evidence_target(obs: Obs, valid_components: Sequence[str], history: List[str]) -> str:
    evidence = evidence_text(obs, history)
    ranked = prioritized_components(obs, valid_components)
    for component in ranked:
        if _norm(component) in evidence:
            return component
    return ranked[0] if ranked else ""


def fallback_candidates(obs: Obs, valid_components: Sequence[str]) -> List[Tuple[str, str]]:
    components = prioritized_components(obs, valid_components)
    kind = incident_kind(obs)
    evidence_text = _norm(f"{obs.goal} {obs.alert_summary} {obs.action_feedback}")
    candidates: List[Tuple[str, str]] = []

    if kind == "data_quality":
        pipeline_components = [c for c in components if "pipeline" in _norm(c)]
        for component in pipeline_components:
            candidates.extend([
                ("query_logs", component),
                ("inspect", component),
            ])
        for component in components:
            if "feature_store" in component:
                candidates.extend([
                    ("check_metrics", component),
                    ("inspect", component),
                ])
        for component in pipeline_components:
            candidates.append(("check_metrics", component))

    if kind == "latency":
        for component in components:
            candidates.extend([
                ("inspect", component),
                ("query_logs", component),
                ("compare_configs", component),
                ("check_metrics", component),
            ])

    if kind == "drift":
        feature_target = next((c for c in components if "feature" in _norm(c)), components[0] if components else "")
        if feature_target:
            candidates.append(("check_feature_drift", feature_target))
        if "business" not in valid_components:
            candidates.append(("check_metrics", "business"))
        for preferred in ["model_server", "training_pipeline", "ab_testing_service", "feature_store"]:
            component = next((c for c in components if preferred in c), None)
            if component:
                candidates.extend([
                    ("inspect", component),
                    ("query_logs", component),
                    ("check_metrics", component),
                ])

    if kind == "cascade":
        primary = [
            c for c in components
            if any(term in _norm(obs.component_status.get(c, "")) for term in ["critical", "error", "degraded"])
            and "model_serving" not in c
            and "model_server" not in c
        ]
        for component in primary:
            candidates.append(("query_logs", component))
        for component in primary:
            candidates.extend([
                ("check_metrics", component),
                ("inspect", component),
            ])
        for component in primary:
            candidates.append(("compare_configs", component))

    for component in components:
        candidates.append(("inspect", component))
        candidates.append(("query_logs", component))
        candidates.append(("check_metrics", component))

    if any(term in evidence_text for term in ["deploy", "config", "change", "rollback", "migration"]):
        for component in components:
            candidates.append(("compare_configs", component))

    for component in components:
        candidates.append(("compare_configs", component))

    return candidates


def prioritized_components(obs: Obs, valid_components: Sequence[str]) -> List[str]:
    kind = incident_kind(obs)

    def rank(component: str) -> Tuple[int, int, int, str]:
        state = _norm(obs.component_status.get(component, ""))
        if any(term in state for term in ["critical", "error", "failed"]):
            status_rank = 0
        elif any(term in state for term in ["degraded", "warning", "warn"]):
            status_rank = 1
        elif "healthy" in state or "ok" in state:
            status_rank = 3
        else:
            status_rank = 2

        role_rank = 5
        name = _norm(component)
        if kind == "data_quality" and "pipeline" in name:
            role_rank = 0
        elif kind == "data_quality" and "feature store" in name:
            role_rank = 1
        elif kind == "drift" and any(term in name for term in ["feature store", "model server", "training pipeline", "ab testing"]):
            role_rank = 0
        elif kind == "latency" and any(term in name for term in ["preprocessor", "model serving", "model server", "load balancer"]):
            role_rank = 0
        elif kind == "cascade" and "model serving" not in name and "model server" not in name:
            role_rank = 0

        mentioned = _norm(component) in _norm(f"{obs.action_feedback} {obs.alert_summary}")
        return (status_rank, role_rank, 0 if mentioned else 1, component)

    return sorted(valid_components, key=rank)


def evidence_summary(obs: Obs, history: List[str]) -> str:
    snippets = [
        f"Incident goal: {obs.goal}",
        f"Alert: {obs.alert_summary}",
    ] + history[-6:] + [f"Latest feedback: {obs.action_feedback}"]
    return "Evidence-based diagnosis from observations: " + " | ".join(s[:700] for s in snippets if s)


def chat_message(role: str, content: str) -> ChatCompletionMessageParam:
    return cast(ChatCompletionMessageParam, {"role": role, "content": content})


def messages_for_model(messages: List[ChatCompletionMessageParam]) -> List[ChatCompletionMessageParam]:
    """Keep local message memory, but send a compact request to small-context models."""
    if not messages:
        return []
    return [messages[0], messages[-1]]


def _append_oracle_trace(
    path: str,
    task_id: str,
    final_score: float,
    episode_success: bool,
    messages: List[ChatCompletionMessageParam],
    final_diagnosis: Optional[Dict[str, Any]],
) -> None:
    """Append one JSON line: full chat + final diagnosis for SFT conversion (hf_train)."""
    if not path or final_score < ORACLE_TRACE_MIN_SCORE:
        return
    if not final_diagnosis or not str(final_diagnosis.get("target", "")).strip():
        return
    tdir = os.path.dirname(os.path.abspath(path))
    if tdir:
        os.makedirs(tdir, exist_ok=True)
    row = {
        "schema": "oracle_trace_v1",
        "task_id": task_id,
        "final_score": final_score,
        "success": episode_success,
        "messages": [{"role": str(m["role"]), "content": str(m.get("content") or "")} for m in messages],
        "final_diagnosis": final_diagnosis,
        "ts": time.time(),
    }
    with open(path, "a", encoding="utf-8") as tf:
        tf.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[TRACE] oracle trace appended → {path}", flush=True)


# Task runner
def run_task(
    llm: OpenAI,
    env: DirectEnv,
    task_id: str,
    *,
    trace_path: Optional[str] = None,
) -> dict:
    rewards: List[float] = []
    final_score = 0.0
    success = False
    history: List[str] = []
    seen: set[str] = set()
    obs = Obs()
    messages: List[ChatCompletionMessageParam] = [chat_message("system", SYSTEM_PROMPT)]
    last_submit: Optional[Dict[str, Any]] = None

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_id=task_id)
        valid_components = list(obs.component_status.keys())
        task_max_steps = TASK_STEP_BUDGET.get(task_id, MAX_STEPS)

        for step in range(1, task_max_steps + 1):
            if obs.done:
                break

            remaining_steps = task_max_steps - step + 1
            user_prompt = build_user_prompt(obs, history, seen, remaining_steps, task_id)
            messages.append(chat_message("user", user_prompt))

            last_error = "null"
            source = "llm"
            parsed: Optional[Tuple[str, str, dict]] = None

            try:
                completion = create_completion_with_retry(llm, messages)
                raw = completion.choices[0].message.content or ""
                messages.append(chat_message("assistant", raw))
                parsed = parse_action(raw, valid_components)
            except Exception as e:
                last_error = str(e).replace("\n", " ")[:160]

            if not parsed:
                action_type, target, parameters = fallback_action(
                    obs, valid_components, seen, history, remaining_steps, task_id
                )
                source = "fallback"
            else:
                action_type, target, parameters = parsed

            forced_submit = should_force_submit(task_id, remaining_steps)
            if forced_submit:
                action_type, target, parameters = fallback_diagnosis(obs, valid_components, history)
                source = "fallback_force_submit_guard"

            if not forced_submit and action_type in BLOCKED_ACTIONS:
                action_type, target, parameters = fallback_investigation_action(
                    obs, valid_components, seen
                )
                source = "fallback_blocked_action_guard"

            if (
                action_type == "check_feature_drift"
                and (
                    incident_kind(obs) != "drift"
                    or not is_allowed_drift_component(target)
                )
            ):
                action_type, target, parameters = fallback_investigation_action(
                    obs, valid_components, seen
                )
                source = "fallback_action_guard"

            if action_type != "submit_diagnosis" and f"{action_type}:{target}" in seen:
                if submit_ready(obs, history, seen, remaining_steps):
                    action_type, target, parameters = fallback_diagnosis(
                        obs, valid_components, history
                    )
                    source = "fallback_loop_submit"
                else:
                    action_type, target, parameters = fallback_action(
                        obs, valid_components, seen, history, remaining_steps, task_id
                    )
                    source = "fallback_loop_guard"

            if (
                not forced_submit
                and action_type == "submit_diagnosis"
                and not submit_ready(obs, history, seen, remaining_steps)
            ):
                action_type, target, parameters = fallback_action(
                    obs, valid_components, seen, history, remaining_steps, task_id
                )
                source = "fallback_submit_guard"

            if not forced_submit and action_type == "submit_diagnosis" and not diagnosis_plausible(
                action_type, target, parameters, obs, history
            ):
                action_type, target, parameters = fallback_diagnosis(obs, valid_components, history)
                source = "fallback_diagnosis_guard"

            seen.add(f"{action_type}:{target}")

            if action_type == "submit_diagnosis":
                last_submit = {
                    "target": target,
                    "root_cause": str(parameters.get("root_cause", "")),
                    "fix": str(parameters.get("fix", "")),
                }

            try:
                obs = env.step(action_type, target, parameters)
            except Exception as e:
                last_error = str(e).replace("\n", " ")[:160]
                print(
                    f"[STEP] step={step} action={action_type}({target}) "
                    f"source={source} reward=0.00 done=false error={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                history.append(f"Step {step}: {action_type}({target}) ERROR {last_error}")
                break

            action_str = f"{action_type}({target})"
            feedback = (obs.action_feedback or "").replace("\n", " ")[:900]
            history.append(
                f"Step {step}: {action_str} source={source} reward={obs.reward:+.2f} feedback={feedback}"
            )
            rewards.append(obs.reward)

            print(
                f"[STEP] step={step} action={action_str} source={source} "
                f"reward={obs.reward:.2f} done={'true' if obs.done else 'false'} "
                f"error={last_error}",
                flush=True,
            )

            final_score = obs.final_score if obs.final_score is not None else obs.cumulative_reward
            if obs.done:
                break

        success = bool(obs.done and final_score > 0.0)

    except Exception as e:
        print(f"# run_task error: {e}", flush=True)
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        final_score = round(min(max(final_score, 0.0001), 0.9999), 4)
        print(
            f"[END] success={'true' if success else 'false'} steps={len(rewards)} "
            f"score={final_score:.4f} rewards={rewards_str}",
            flush=True,
        )
        if trace_path:
            _append_oracle_trace(trace_path, task_id, final_score, success, messages, last_submit)

    return {"task_id": task_id, "final_score": final_score, "score_breakdown": obs.score_breakdown}


# Entry point
def main() -> None:
    if not API_KEY:
        print("# WARNING: No API key set", flush=True)

    env = DirectEnv(base_url=HF_SPACE_URL)
    if not env.health():
        print(f"# WARNING: Cannot reach {HF_SPACE_URL}", flush=True)

    print(f"# Server: {HF_SPACE_URL}", flush=True)
    print(f"# Model:  {MODEL_NAME}", flush=True)
    print(f"# Temperature: {TEMPERATURE}", flush=True)
    if ORACLE_TRACE_PATH:
        print(
            f"# Oracle traces: ORACLE_TRACE_PATH={ORACLE_TRACE_PATH} "
            f"(min_score={ORACLE_TRACE_MIN_SCORE})",
            flush=True,
        )

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key", timeout=30)
    results = []
    task_ids = resolve_tasks()
    trace_path = ORACLE_TRACE_PATH if ORACLE_TRACE_PATH else None

    print(f"# Tasks: {', '.join(task_ids)}", flush=True)

    for task_id in task_ids:
        results.append(run_task(llm, env, task_id, trace_path=trace_path))

    print(f"\n# {'=' * 58}", flush=True)
    print("# LLM AGENT RESULTS", flush=True)
    print(f"# {'=' * 58}", flush=True)
    total = 0.0
    for result in results:
        print(f"#  {result['task_id']:<10} score: {result['final_score']:.4f}", flush=True)
        total += result["final_score"]
    print(f"#  {'AVERAGE':<10} score: {total / len(results):.4f}", flush=True)


if __name__ == "__main__":
    main()
