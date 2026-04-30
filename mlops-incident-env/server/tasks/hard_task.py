"""
hard_task.py — Grader v2 for Task 3: Silent Model Drift
Deep multi-criteria: drift ID + root cause chain + business impact + remediation plan + process depth
No shortcuts — must demonstrate full investigation chain to score high
"""
from __future__ import annotations
import re
from typing import Any, Dict, List

from .grading_utils import contains_any, normalize_text


class HardTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None,
              root_cause_keywords: List[str] | None = None,
              broken_component: str | None = None,
              **_: Any) -> Dict[str, Any]:
        text = normalize_text(str(diagnosis_target) + " " + str(diagnosis_params))
        path = [normalize_text(p) for p in (investigation_path or [])]
        keywords = root_cause_keywords or []
        drifted_feature = broken_component or (keywords[0] if keywords else "user_engagement_score")
        experiment = next(
            (kw for kw in keywords if re.fullmatch(r"[a-z]\d+", normalize_text(kw))),
            "experiment",
        )
        model_age = next(
            (kw for kw in keywords if normalize_text(kw).endswith("days")),
            "model age",
        )
        score = 0.0
        breakdown: Dict[str, float] = {}

        # 1. Drift identification (0.30)
        feature_terms = _feature_terms(drifted_feature)
        feature_score = 0.20 if contains_any(text, feature_terms) else 0.0
        psi_score = 0.10 if contains_any(text, ["psi", "population stability", "distribution shift", "covariate"]) else 0.0
        breakdown["identifies_correct_drifted_feature"] = feature_score
        breakdown["mentions_psi_evidence"] = psi_score
        score += feature_score + psi_score

        # 2. Root cause chain (0.30)
        experiment_score = 0.15 if contains_any(
            text,
            [experiment, "ab test", "experiment", "redesign", "interface change", "checkout", "pricing"],
        ) else 0.0
        model_age_terms = [model_age, normalize_text(model_age).replace(" days", " day")]
        stale_score = 0.15 if contains_any(
            text,
            model_age_terms + ["stale", "not retrained", "outdated model", "old model", "model age"],
        ) else 0.0
        breakdown["connects_experiment_or_product_change"] = experiment_score
        breakdown["mentions_model_staleness"] = stale_score
        score += experiment_score + stale_score

        # 3. Business impact (0.15)
        biz_score = 0.08 if contains_any(text, ["revenue", "business impact", "conversion", "downstream"]) else 0.0
        quant_score = 0.07 if contains_any(text, ["percent", "3 day", "three day", "gradual", "silent"]) or "%" in text else 0.0
        breakdown["connects_drift_to_revenue"] = biz_score
        breakdown["quantifies_impact"] = quant_score
        score += biz_score + quant_score

        # 4. Remediation plan (0.15) — all 3 needed for full marks
        retrain_score = 0.05 if contains_any(text, ["retrain", "fine tune", "fine-tune", "update model"]) else 0.0
        window_score = 0.05 if contains_any(text, ["data window", "training window", "cutoff", "post change", "post redesign", "recent data"]) else 0.0
        monitor_score = 0.05 if contains_any(text, ["monitor", "drift detection", "psi threshold", "continuous", "schedule"]) else 0.0
        breakdown["mentions_retraining"] = retrain_score
        breakdown["specifies_data_window"] = window_score
        breakdown["proposes_ongoing_monitoring"] = monitor_score
        score += retrain_score + window_score + monitor_score

        # 5. Investigation depth (0.10)
        drift_used = 0.05 if any("checkfeaturedrift" in p for p in path) else 0.0
        biz_checked = 0.05 if any("checkmetrics" in p for p in path) else 0.0
        breakdown["used_checkfeaturedrift_action"] = drift_used
        breakdown["checked_metrics_during_investigation"] = biz_checked
        score += drift_used + biz_checked

        # 6. Red herring / wrong direction penalties
        infra_wrong = contains_any(text, ["oom", "out of memory", "latency spike", "pipeline error", "schema mismatch"])
        if infra_wrong and feature_score == 0.0:
            breakdown["wrong_direction_infra_penalty"] = -0.20
            score -= 0.20
        else:
            breakdown["wrong_direction_infra_penalty"] = 0.0

        wrong_feat = contains_any(text, ["transaction amount", "click through rate", "page load", "api response time"])
        if wrong_feat and feature_score == 0.0:
            breakdown["wrong_feature_penalty"] = -0.10
            score -= 0.10
        else:
            breakdown["wrong_feature_penalty"] = 0.0

        return {"total": round(min(max(score, 0.0001), 0.9999), 4), "breakdown": breakdown}


def _feature_terms(feature: str) -> List[str]:
    feature_text = normalize_text(feature)
    terms = [feature_text]

    words = feature_text.split()
    if len(words) > 2:
        terms.append(" ".join(words[1:]))
    if len(words) > 1 and words[-1] == "score":
        terms.append(" ".join(words[:-1]))

    if feature_text.startswith("avg "):
        terms.append("average " + feature_text[4:])

    if "30d" in feature_text:
        terms.extend([
            feature_text.replace("30d", "30 day"),
            feature_text.replace(" 30d", ""),
        ])

    return terms
