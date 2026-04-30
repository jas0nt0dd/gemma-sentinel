"""
medium_task.py — Grader v2 for Task 2: Latency Spike Investigation
Multi-criteria: component ID + config depth + fix quality + process bonus + penalties
"""
from __future__ import annotations
from typing import Any, Dict, List

from .grading_utils import breakdown_label, contains_any, contains_term, normalize_text


class MediumTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None,
              root_cause_keywords: List[str] | None = None,
              broken_component: str | None = None,
              config_diff: Dict[str, Any] | None = None,
              **_: Any) -> Dict[str, Any]:
        text = normalize_text(str(diagnosis_target) + " " + str(diagnosis_params))
        path = [normalize_text(p) for p in (investigation_path or [])]
        keywords = root_cause_keywords or []
        component = broken_component or (keywords[0] if keywords else "feature_preprocessor_v2")
        config_param = keywords[1] if len(keywords) > 1 else "batch_size"
        new_value = keywords[2] if len(keywords) > 2 else "512"
        old_value = config_diff.get("old_value") if config_diff else None
        score = 0.0
        breakdown: Dict[str, float] = {}

        # Gate: component identification
        component_terms = [component]
        component_words = normalize_text(component).split()
        if len(component_words) >= 2:
            component_terms.append(" ".join(component_words[-2:]))

        if contains_any(text, component_terms):
            breakdown["correct_component"] = 0.40
            score += 0.40
        elif component_words and any(part in text for part in component_words if len(part) > 2):
            breakdown["correct_component"] = 0.15  # partial
            score += 0.15
        else:
            breakdown["correct_component"] = 0.0

        # Root cause depth
        has_correct = breakdown["correct_component"] > 0
        param_score = 0.20 if contains_term(text, config_param) else 0.0
        value_score = 0.15 if contains_term(text, new_value) else 0.0
        breakdown["config_parameter_identified"] = param_score
        breakdown["config_value_identified"] = value_score
        score += param_score + value_score

        # Fix quality
        rollback_score = 0.08 if contains_any(text, ["rollback", "revert", "restore", "previous config"]) else 0.0
        previous_value_score = (
            0.07
            if old_value is not None
            and contains_term(text, old_value)
            and contains_term(text, config_param)
            else 0.0
        )
        breakdown["proposes_rollback"] = rollback_score
        breakdown[f"mentions_previous_{breakdown_label(config_param)}"] = previous_value_score
        score += rollback_score + previous_value_score

        # Investigation process bonus
        cmp_bonus = 0.05 if any("compareconfigs" in p for p in path) else 0.0
        met_bonus = 0.05 if any("checkmetrics" in p and contains_term(p, component) for p in path) else 0.0
        breakdown["used_compareconfigs"] = cmp_bonus
        breakdown["checked_metrics"] = met_bonus
        score += cmp_bonus + met_bonus

        # Red herring penalty (only if wrong component identified)
        if not has_correct:
            ms_pen = -0.15 if contains_any(text, ["model server", "model_serving"]) else 0.0
            ag_pen = -0.10 if contains_term(text, "api_gateway") else 0.0
            breakdown["model_server_blame_penalty"] = ms_pen
            breakdown["api_gateway_blame_penalty"] = ag_pen
            score += ms_pen + ag_pen
        else:
            breakdown["model_server_blame_penalty"] = 0.0
            breakdown["api_gateway_blame_penalty"] = 0.0

        # Efficiency bonus
        eff = 0.10 if (step_count and step_count <= 6) else (0.05 if (step_count and step_count <= 9) else 0.0)
        breakdown["efficiency_bonus"] = round(eff, 4)
        score += eff

        return {"total": round(min(max(score, 0.0001), 0.9999), 4), "breakdown": breakdown}
