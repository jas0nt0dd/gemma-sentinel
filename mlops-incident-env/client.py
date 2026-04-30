"""
client.py — HTTPEnvClient for MLOps Incident Response Environment.
"""
from __future__ import annotations
from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import MLOpsAction, MLOpsObservation, MLOpsState


class MLOpsEnv(EnvClient[MLOpsAction, MLOpsObservation, MLOpsState]):
    """
    Usage:
        env = MLOpsEnv(base_url="https://your-space.hf.space")
        obs = env.reset(task_id="easy")
        result = env.step(MLOpsAction(action_type="query_logs", target="data_pipeline_b"))
        result = env.step(MLOpsAction(
            action_type="submit_diagnosis",
            target="data_pipeline_b",
            parameters={"root_cause": "schema_mismatch", "fix": "revert schema migration"}
        ))
    """

    def _step_payload(self, action: MLOpsAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "target": action.target,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs = MLOpsObservation(
            goal=payload.get("goal", ""),
            alert_summary=payload.get("alert_summary", ""),
            component_status=payload.get("component_status", {}),
            recent_logs=payload.get("recent_logs", []),
            metrics_snapshot=payload.get("metrics_snapshot", {}),
            action_feedback=payload.get("action_feedback", ""),
            step_count=payload.get("step_count", 0),
            reward=payload.get("reward", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            final_score=payload.get("final_score"),
            score_breakdown=payload.get("score_breakdown"),
        )
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload: Dict[str, Any]) -> MLOpsState:
        return MLOpsState(
            episode_id=payload.get("episode_id", ""),
            task_id=payload.get("task_id", ""),
            investigation_path=payload.get("investigation_path", []),
            partial_score=payload.get("partial_score", 0.0),
            step_count=payload.get("step_count", 0),
            max_steps=payload.get("max_steps", 20),
            diagnosis_submitted=payload.get("diagnosis_submitted", False),
        )