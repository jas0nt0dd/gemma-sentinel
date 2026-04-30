"""
easy_task.py — Grader v2 for Task 1: Data Quality Alert
Process-gated scoring: investigation quality + efficiency bonus + red herring penalty
"""
from __future__ import annotations
from typing import Any, Dict, List

from .grading_utils import breakdown_label, contains_term, normalize_text


class EasyTaskGrader:
    RED_HERRINGS = {"api_gateway": -0.15, "data_pipeline_a": -0.10}

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None,
              root_cause_keywords: List[str] | None = None,
              broken_component: str | None = None,
              **_: Any) -> Dict[str, Any]:
        text = normalize_text(str(diagnosis_target) + " " + str(diagnosis_params))
        path = [normalize_text(p) for p in (investigation_path or [])]
        keywords = root_cause_keywords or []
        broken = broken_component or (keywords[0] if keywords else "data_pipeline_b")
        bonus_keywords = keywords[1:] if len(keywords) > 1 else ["schema", "migration", "transaction_amount"]
        broken_label = breakdown_label(broken)
        score = 0.0
        breakdown: Dict[str, float] = {}

        # Gate: correct component (required for bonus points)
        if contains_term(text, broken):
            breakdown[f"identified_{broken_label}"] = 0.40
            score += 0.40
            # Bonus keywords only count if gate passed
            for kw in bonus_keywords:
                pts = 0.10
                key = f"mentioned_{breakdown_label(kw)}"
                if contains_term(text, kw):
                    breakdown[key] = pts
                    score += pts
                else:
                    breakdown[key] = 0.0
        else:
            breakdown[f"identified_{broken_label}"] = 0.0
            for kw in bonus_keywords:
                breakdown[f"mentioned_{breakdown_label(kw)}"] = 0.0
            # Red herring penalties only if wrong answer
            for rh, pen in self.RED_HERRINGS.items():
                if rh != broken and contains_term(text, rh):
                    breakdown[f"blamed_{rh}_penalty"] = pen
                    score += pen
                else:
                    breakdown[f"blamed_{rh}_penalty"] = 0.0

        # Investigation quality bonus
        iq = 0.0
        iq += 0.10 if any("querylogs" in p and contains_term(p, broken) for p in path) else 0.0
        iq += 0.08 if any("checkmetrics" in p and "feature" in p for p in path) else 0.0
        iq += 0.07 if any("inspect" in p and contains_term(p, broken) for p in path) else 0.0
        breakdown["investigation_quality_bonus"] = round(iq, 4)
        score += iq

        # Efficiency bonus
        eff = 0.15 if (step_count and step_count <= 4) else (0.08 if (step_count and step_count <= 6) else (0.03 if (step_count and step_count <= 8) else 0.0))
        breakdown["efficiency_bonus"] = round(eff, 4)
        score += eff

        return {"total": round(min(max(score, 0.0001), 0.9999), 4), "breakdown": breakdown}
