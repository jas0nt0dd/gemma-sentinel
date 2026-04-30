"""
environment.py — MLOps Incident Response Environment v2
Upgrades over v1:
  - ScenarioGenerator replaces static JSON loading (dynamic incidents every episode)
  - Graders receive investigation_path + step_count for process-gated scoring
  - Red herring components in all scenarios mislead weak agents
  - SLA countdown visible in observation (urgency pressure)
  - Cascade failure task (4th task) — multi-root-cause requiring synthesis
  - Step efficiency tracked and passed to graders

Innovation / teaching signal (hackathon “environment” judging):
  - ground_truth per scenario: primary vs related components, red herrings, incident_pattern
  - Investigation steps: higher reward for probing primary/root, medium for related, low/zero for
    irrelevant logs; compare_configs rewards matching the true changed service
  - Action–pattern bonuses (e.g. check_feature_drift when pattern is feature_drift)
  - Repeat probes on red-herring components incur a small penalty
  - Successful submit_diagnosis gets an efficiency bonus when step_count is below an ideal budget
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scenario_generator import ScenarioGenerator

_generator = ScenarioGenerator()

TASK_MAX_STEPS: Dict[str, int] = {
    "easy": 10,
    "medium": 15,
    "hard": 25,
    "cascade": 30,
}

VALID_ACTIONS = [
    "inspect", "querylogs", "checkmetrics",
    "compareconfigs", "checkfeaturedrift",
    "submitdiagnosis", "requestrollback",
]


@dataclass
class ObsPayload:
    goal: str
    alert_summary: str
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "alert_summary": self.alert_summary,
            "component_status": self.component_status,
            "recent_logs": self.recent_logs,
            "metrics_snapshot": self.metrics_snapshot,
            "action_feedback": self.action_feedback,
            "step_count": self.step_count,
            "sla_minutes_remaining": self.sla_minutes_remaining,
            "reward": self.reward,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "final_score": self.final_score,
            "score_breakdown": self.score_breakdown,
        }


@dataclass
class EpisodeState:
    episode_id: str
    task_id: str
    true_root_cause: str
    root_cause_keywords: List[str] = field(default_factory=list)
    investigation_path: List[str] = field(default_factory=list)
    partial_score: float = 0.0
    step_count: int = 0
    max_steps: int = 20
    sla_steps: Optional[int] = None
    diagnosis_submitted: bool = False
    relevant_components_visited: List[str] = field(default_factory=list)
    red_herring_visits: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "investigation_path": self.investigation_path,
            "partial_score": self.partial_score,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "diagnosis_submitted": self.diagnosis_submitted,
            "relevant_components_visited": self.relevant_components_visited,
        }


class MLOpsEnvironment:
    """Simulates a production ML system under incident conditions."""

    def __init__(self) -> None:
        self.scenario: Dict[str, Any] = {}
        self.state: EpisodeState = EpisodeState("", "", "")
        self.cumulative_reward: float = 0.0
        self.seen_actions: set = set()

    # ─── Public OpenEnv interface ────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> ObsPayload:
        if task_id not in TASK_MAX_STEPS:
            task_id = "easy"

        # ── Dynamic scenario (replaces static JSON load) ──────────────────
        self.scenario = _generator.generate(task_id)

        self.state = EpisodeState(
            episode_id=str(uuid.uuid4())[:8],
            task_id=task_id,
            true_root_cause=self.scenario["root_cause"],
            root_cause_keywords=self.scenario.get("root_cause_keywords", []),
            investigation_path=[],
            partial_score=0.0,
            step_count=0,
            max_steps=TASK_MAX_STEPS[task_id],
            sla_steps=self.scenario.get("sla_steps"),
            diagnosis_submitted=False,
            relevant_components_visited=[],
            red_herring_visits={},
        )
        self.cumulative_reward = 0.0
        self.seen_actions = set()

        components = self.scenario.get("components", {})
        return ObsPayload(
            goal=self.scenario["goal"],
            alert_summary=self.scenario["alert_summary"],
            component_status={k: v["status"] for k, v in components.items()},
            recent_logs=[],
            metrics_snapshot=self.scenario.get("global_metrics", {}),
            action_feedback=(
                f"Incident opened. Begin your investigation.\n"
                f"Available components: {', '.join(components.keys())}\n"
                f"Available actions: {', '.join(VALID_ACTIONS)}"
            ),
            step_count=0,
            sla_minutes_remaining=self.scenario.get("sla_minutes"),
            reward=0.0,
            cumulative_reward=0.0,
            done=False,
        )

    def step(self, action_type: str, target: str, parameters: Dict[str, Any]) -> ObsPayload:
        # Normalize: accept both query_logs and querylogs, submit_diagnosis and submitdiagnosis
        action_type = action_type.replace("_", "").replace("-", "").lower()
        self.state.step_count += 1
        action_key = f"{action_type}:{target}"
        self.state.investigation_path.append(action_key)

        reward = 0.0
        done = False
        feedback = ""
        logs: List[Dict] = []
        metrics: Dict = {}
        final_score = None
        score_breakdown = None
        reward_cfg = self.scenario.get("reward_config", {})

        # SLA countdown
        sla_remaining = None
        if self.scenario.get("sla_minutes") and self.state.sla_steps:
            steps_left = self.state.sla_steps - self.state.step_count
            sla_remaining = max(0, steps_left * 2)

        # Loop detection
        if action_key in self.seen_actions and action_type != "submitdiagnosis":
            reward = reward_cfg.get("loop_penalty", -0.15)
            feedback = (
                f"⚠ Already ran {action_type}:{target}. No new info. "
                f"Explore a different component."
            )
        else:
            self.seen_actions.add(action_key)
            if action_type == "inspect":
                reward, feedback, logs, metrics = self._do_inspect(target, reward_cfg)
            elif action_type == "querylogs":
                reward, feedback, logs, metrics = self._do_query_logs(target, reward_cfg)
            elif action_type == "checkmetrics":
                reward, feedback, logs, metrics = self._do_check_metrics(target, reward_cfg)
            elif action_type == "compareconfigs":
                reward, feedback, logs, metrics = self._do_compare_configs(target, reward_cfg)
            elif action_type == "checkfeaturedrift":
                reward, feedback, logs, metrics = self._do_feature_drift(reward_cfg)
            elif action_type == "submitdiagnosis":
                reward, feedback, done, final_score, score_breakdown = self._do_diagnosis(
                    target, parameters, reward_cfg
                )
            elif action_type == "requestrollback":
                reward, feedback = self._do_rollback(target)
            else:
                feedback = f"Unknown action '{action_type}'. Valid: {', '.join(VALID_ACTIONS)}"

        # Step limit
        if self.state.step_count >= self.state.max_steps and not done:
            reward += reward_cfg.get("step_over_max_penalty", -0.05)
            done = True
            feedback += f"\n⏰ Max steps ({self.state.max_steps}) reached. Episode ending."

        self.cumulative_reward += reward
        self.state.partial_score = round(self.cumulative_reward, 4)

        components = self.scenario.get("components", {})
        return ObsPayload(
            goal=self.scenario["goal"],
            alert_summary=self.scenario["alert_summary"],
            component_status={k: v["status"] for k, v in components.items()},
            recent_logs=logs,
            metrics_snapshot=metrics if metrics else self.scenario.get("global_metrics", {}),
            action_feedback=feedback,
            step_count=self.state.step_count,
            sla_minutes_remaining=sla_remaining,
            reward=round(reward, 4),
            cumulative_reward=round(self.cumulative_reward, 4),
            done=done,
            final_score=final_score,
            score_breakdown=score_breakdown,
        )

    @property
    def state_info(self) -> EpisodeState:
        return self.state

    # ─── Ground-truth–aware shaping (richer mid-episode signal for agents / RL) ─

    def _ground_truth(self) -> Dict[str, Any]:
        return self.scenario.get("ground_truth") or {}

    def _apply_red_herring_shaping(self, target: str, reward: float) -> Tuple[float, str]:
        rh = list(self._ground_truth().get("red_herrings") or [])
        if target not in rh:
            return reward, ""
        self.state.red_herring_visits[target] = self.state.red_herring_visits.get(target, 0) + 1
        n = self.state.red_herring_visits[target]
        if n <= 1:
            return reward, ""
        pen = 0.05
        return reward - pen, f"\n⚠ Repeated probe on distractor '{target}' (−{pen}). Focus on root-cause components."

    def _action_pattern_bonus(self, action_norm: str) -> float:
        pattern = self._ground_truth().get("incident_pattern") or ""
        table: Dict[str, Dict[str, float]] = {
            "feature_drift": {"checkfeaturedrift": 0.08, "checkmetrics": 0.04, "inspect": 0.02, "querylogs": 0.02},
            "config_regression": {"compareconfigs": 0.08, "inspect": 0.05, "querylogs": 0.04, "checkmetrics": 0.04},
            "schema_drift": {"querylogs": 0.08, "checkmetrics": 0.04, "inspect": 0.04, "compareconfigs": 0.02},
            "cascade_multi": {"checkmetrics": 0.06, "querylogs": 0.05, "inspect": 0.04, "compareconfigs": 0.03},
        }
        return table.get(pattern, {}).get(action_norm, 0.0)

    def _efficiency_bonus(self, task_id: str, steps_at_submit: int) -> float:
        """Reward shorter successful investigations (caps in grader total)."""
        ideal_coef = {
            "easy": (5, 0.02),
            "medium": (7, 0.015),
            "hard": (9, 0.01),
            "cascade": (12, 0.008),
        }
        ideal, coef = ideal_coef.get(task_id, (8, 0.01))
        return max(0.0, float(ideal - steps_at_submit) * coef)

    # ─── Action handlers ─────────────────────────────────────────────────────

    def _do_inspect(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self.scenario.get("components", {})
        if target not in components:
            return (
                0.0,
                f"Unknown component '{target}'. Available: {', '.join(components.keys())}",
                [], {},
            )
        comp = components[target]
        reward = 0.0
        relevant = reward_cfg.get("relevant_components", [])
        gt = self._ground_truth()
        primary = gt.get("primary_component")
        if target in relevant and target not in self.state.relevant_components_visited:
            base = reward_cfg.get("explore_relevant_reward", 0.05)
            reward = base * 1.55 if primary and target == primary else base
            self.state.relevant_components_visited.append(target)

        reward += self._action_pattern_bonus("inspect")
        reward, rh_note = self._apply_red_herring_shaping(target, reward)

        has_config = (
            "config_diff" in self.scenario
            and self.scenario["config_diff"].get("service") == target
        )
        has_drift = "feature_drift_data" in self.scenario
        extra = ", compare_configs" if has_config else (", check_feature_drift" if has_drift else "")

        feedback = (
            f"COMPONENT: {target.upper()}\n"
            f"Status: {comp['status'].upper()}\n"
            f"Description: {comp['description']}\n"
            f"Last updated: {comp.get('last_updated', 'unknown')}\n"
            f"Next steps: query_logs{extra}, check_metrics"
        )
        if rh_note:
            feedback += rh_note
        logs = comp.get("logs", [])
        metrics = comp.get("metrics", {})
        return reward, feedback, logs, metrics

    def _do_query_logs(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self.scenario.get("components", {})
        if target not in components:
            return (
                0.0,
                f"Unknown component '{target}'. Available: {', '.join(components.keys())}",
                [], {},
            )
        comp = components[target]
        logs = comp.get("logs", [])
        reward = 0.0

        key_evidence = list(reward_cfg.get("key_evidence_logs", []) or [])
        fe = float(reward_cfg.get("find_key_evidence_reward", 0.10))
        gt = self._ground_truth()
        primary = gt.get("primary_component")
        related = set(gt.get("related_components") or [])

        if primary and target == primary:
            reward = fe
        elif target in related:
            reward = fe * 0.58
        elif target in key_evidence:
            reward = fe * 0.42
        elif not primary and target in key_evidence:
            reward = fe

        reward += self._action_pattern_bonus("querylogs")
        rh_note = ""
        reward, rh_note = self._apply_red_herring_shaping(target, reward)

        if not logs:
            feedback = f"No logs available for {target}."
        else:
            lines = "\n".join(
                f"[{e.get('time','?')}] {e.get('level','INFO')}: {e.get('msg', e.get('message',''))}"
                for e in logs
            )
            feedback = f"LOGS — {target.upper()}:\n{lines}"
        if rh_note:
            feedback += rh_note

        return reward, feedback, logs, {}

    def _do_check_metrics(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        components = self.scenario.get("components", {})

        # Allow "global" or "business" as special targets
        if target in ("global", "business", "revenue"):
            biz = self.scenario.get("business_metrics", {})
            global_m = self.scenario.get("global_metrics", {})
            combined = {**global_m, **biz}
            reward = float(reward_cfg.get("find_business_metrics_reward", 0.10) if biz else 0.0)
            reward += self._action_pattern_bonus("checkmetrics")
            feedback = f"GLOBAL METRICS:\n{self._fmt_dict(combined)}"
            return reward, feedback, [], combined

        if target not in components:
            return (
                0.0,
                f"Unknown component '{target}'. Available: {', '.join(components.keys())}",
                [], {},
            )
        comp = components[target]
        metrics = comp.get("metrics", {})
        reward = 0.0

        relevant = reward_cfg.get("relevant_components", [])
        base = float(reward_cfg.get("explore_relevant_reward", 0.04))
        gt = self._ground_truth()
        primary = gt.get("primary_component")
        related = set(gt.get("related_components") or [])
        if target == primary:
            reward = base * 1.45
        elif target in related:
            reward = base * 0.9
        elif target in relevant:
            reward = base * 0.65

        reward += self._action_pattern_bonus("checkmetrics")
        rh_note = ""
        reward, rh_note = self._apply_red_herring_shaping(target, reward)

        feedback = f"METRICS — {target.upper()}:\n{self._fmt_dict(metrics)}"
        if rh_note:
            feedback += rh_note
        return reward, feedback, [], metrics

    def _do_compare_configs(self, target: str, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        config_diff = self.scenario.get("config_diff")
        if not config_diff:
            return (
                0.0,
                "No config changes recorded for this incident. Try query_logs or check_metrics.",
                [], {},
            )
        svc = str(config_diff.get("service") or "")
        reward = 0.12 if target == svc else 0.02
        reward += self._action_pattern_bonus("compareconfigs")
        rh_note = ""
        reward, rh_note = self._apply_red_herring_shaping(target, reward)
        diff = config_diff
        feedback = (
            f"CONFIG DIFF — {diff.get('service', target).upper()}\n"
            f"Parameter:  {diff.get('parameter', 'unknown')}\n"
            f"Old value:  {diff.get('old_value', '?')}\n"
            f"New value:  {diff.get('new_value', '?')}\n"
            f"Changed by: {diff.get('changed_by', 'unknown')}\n"
            f"Changed at: {diff.get('changed_at', 'unknown')}\n"
            f"Summary:    {diff.get('diff_summary', '')}"
        )
        if target != svc and svc:
            feedback += f"\n⚠ You queried compare_configs for '{target}' but the recorded change is on '{svc}'."
        if rh_note:
            feedback += rh_note
        return reward, feedback, [], diff

    def _do_feature_drift(self, reward_cfg: Dict) -> Tuple[float, str, List, Dict]:
        drift_data = self.scenario.get("feature_drift_data")
        if not drift_data:
            return (
                0.0,
                "No feature drift data available for this incident. Try inspect or query_logs.",
                [], {},
            )
        features = drift_data.get("features", {})
        critical = {k: v for k, v in features.items() if v.get("status") == "CRITICAL_DRIFT"}
        reward = reward_cfg.get("find_feature_drift_reward", 0.15)
        if critical:
            reward = reward_cfg.get("find_critical_feature_reward", 0.20)

        reward += self._action_pattern_bonus("checkfeaturedrift")

        lines = []
        for fname, fdata in features.items():
            status = fdata.get("status", "unknown")
            psi = fdata.get("psi", "N/A")
            note = fdata.get("note", "")
            lines.append(f"  {fname}: PSI={psi} [{status}] — {note}")

        feedback = (
            f"FEATURE DRIFT REPORT\n"
            f"{drift_data.get('summary', '')}\n\n"
            + "\n".join(lines)
        )
        return reward, feedback, [], features

    def _do_rollback(self, target: str) -> Tuple[float, str]:
        reward = 0.05
        feedback = (
            f"ROLLBACK INITIATED for {target}.\n"
            f"Rolling back to previous stable configuration.\n"
            f"Monitor system metrics after rollback completes (~2 min).\n"
            f"Submit your diagnosis to close the incident."
        )
        return reward, feedback

    def _do_diagnosis(
        self,
        target: str,
        parameters: Dict[str, Any],
        reward_cfg: Dict,
    ) -> Tuple[float, str, bool, float, Dict]:
        if self.state.diagnosis_submitted:
            return (
                -0.05,
                "Diagnosis already submitted for this episode.",
                True, self.state.partial_score, {},
            )
        self.state.diagnosis_submitted = True

        # Route to the correct task grader
        task_id = self.state.task_id
        root_cause_keywords = self.state.root_cause_keywords
        grader_context = {
            "root_cause_keywords": root_cause_keywords,
            "broken_component": root_cause_keywords[0] if root_cause_keywords else None,
            "config_diff": self.scenario.get("config_diff"),
        }
        if task_id == "easy":
            from tasks.easy_task import EasyTaskGrader
            result = EasyTaskGrader().grade(
                target, parameters,
                self.state.investigation_path,
                self.state.step_count,
                **grader_context,
            )
        elif task_id == "medium":
            from tasks.medium_task import MediumTaskGrader
            result = MediumTaskGrader().grade(
                target, parameters,
                self.state.investigation_path,
                self.state.step_count,
                **grader_context,
            )
        elif task_id == "hard":
            from tasks.hard_task import HardTaskGrader
            result = HardTaskGrader().grade(
                target, parameters,
                self.state.investigation_path,
                self.state.step_count,
                **grader_context,
            )
        elif task_id == "cascade":
            from tasks.cascade_task import CascadeTaskGrader
            result = CascadeTaskGrader().grade(
                target, parameters,
                self.state.investigation_path,
                self.state.step_count,
                **grader_context,
            )
        else:
            result = {"total": 0.0, "breakdown": {}}

        final_score = result["total"]
        breakdown = dict(result.get("breakdown", {}))
        eff = self._efficiency_bonus(self.state.task_id, self.state.step_count)
        final_score = round(min(1.0, float(final_score) + eff), 4)
        if eff > 0:
            breakdown["efficiency_bonus"] = round(eff, 4)
        reward = final_score

        if final_score >= 0.85:
            verdict = "✅ EXCELLENT — Full root cause identified with strong evidence."
        elif final_score >= 0.60:
            verdict = "🟡 GOOD — Correct direction but missing key details."
        elif final_score >= 0.30:
            verdict = "🟠 PARTIAL — Some relevant findings but incomplete diagnosis."
        else:
            verdict = "❌ INCORRECT — Root cause not identified."

        feedback = (
            f"DIAGNOSIS SUBMITTED\n"
            f"Final Score: {final_score:.4f}\n"
            f"{verdict}\n\n"
            f"Score Breakdown:\n"
            + "\n".join(f"  {k}: {v}" for k, v in breakdown.items())
        )
        return reward, feedback, True, final_score, breakdown

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_dict(d: Dict, indent: int = 2) -> str:
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{' '*indent}{k}:")
                lines.append(MLOpsEnvironment._fmt_dict(v, indent + 2))
            else:
                lines.append(f"{' '*indent}{k}: {v}")
        return "\n".join(lines)
