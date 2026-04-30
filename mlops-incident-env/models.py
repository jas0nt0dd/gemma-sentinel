"""
models.py — Typed dataclasses for MLOps Incident Response Environment.
Action / Observation / State following the OpenEnv spec.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from openenv.core.env_server import Action, Observation, State


# ── Action ────────────────────────────────────────────────────────────────────
@dataclass
class MLOpsAction(Action):
    """
    action_type choices:
      inspect             — view component overview & current status
      query_logs          — filter logs for a specific component
      check_metrics       — get time-series metrics for a component
      compare_configs     — diff current vs previous deployment config
      check_feature_drift — view feature distribution & PSI drift scores
      submit_diagnosis    — FINAL answer: root_cause + remediation_plan
      request_rollback    — trigger rollback of the last deployment
    """
    action_type: str = "inspect"
    target: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


# ── Observation ───────────────────────────────────────────────────────────────
@dataclass
class MLOpsObservation(Observation):
    goal: str = ""
    alert_summary: str = ""
    component_status: Dict[str, str] = field(default_factory=dict)
    recent_logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    action_feedback: str = ""
    step_count: int = 0
    reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    final_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None


# ── State (server-side — NEVER sent to agent in full) ─────────────────────────
@dataclass
class MLOpsState(State):
    episode_id: str = ""
    task_id: str = ""
    true_root_cause: str = ""           # Hidden — used only by graders
    root_cause_keywords: List[str] = field(default_factory=list)
    investigation_path: List[str] = field(default_factory=list)
    partial_score: float = 0.0
    step_count: int = 0
    max_steps: int = 20
    diagnosis_submitted: bool = False
    relevant_components_visited: List[str] = field(default_factory=list)