"""
cascade_task.py — Grader for Task 4: Cascade Failure (ELITE difficulty)

Scenario: 3 simultaneous failures triggered by a single deployment
  Root causes (ALL must be identified for full score):
    1. embedding_service_v3 — ONNX runtime version mismatch → corrupted embeddings
    2. feature_store         — cache TTL set to 0s → serving stale features to ALL models
    3. ab_test_router        — traffic split config corrupted → 100% traffic to model B (untested)

  This task simulates a real production cascade where one bad deploy causes 3 downstream failures.
  An agent must synthesize evidence across ALL 3 subsystems to score > 0.7.

Scoring (max 1.0) — must demonstrate cross-system synthesis:
  Identifies embedding_service issue        +0.20
  Identifies feature_store cache issue      +0.20
  Identifies ab_test_router misconfiguration +0.20
  Connects all 3 to single deployment       +0.15
  Proposes coordinated rollback plan        +0.15
  Investigation covered all 3 subsystems    +0.10 (process bonus)

  Partial penalties:
  - Only finds 1 root cause                 capped at 0.35
  - Only finds 2 root causes                capped at 0.70
"""
from __future__ import annotations
from typing import Any, Dict, List

from .grading_utils import breakdown_label, contains_any, contains_term, normalize_text


class CascadeTaskGrader:

    def grade(self, diagnosis_target: str, diagnosis_params: Dict[str, Any],
              investigation_path: List[str] | None = None, step_count: int | None = None,
              root_cause_keywords: List[str] | None = None,
              broken_component: str | None = None,
              **_: Any) -> Dict[str, Any]:
        text = normalize_text(str(diagnosis_target) + " " + str(diagnosis_params))
        path = [normalize_text(p) for p in (investigation_path or [])]
        keywords = root_cause_keywords or []
        cause_services = keywords[:3] if len(keywords) >= 3 else [
            "embedding_service_v3",
            "feature_store",
            "ab_test_router",
        ]
        deployment = keywords[4] if len(keywords) > 4 else "deployment"
        score = 0.0
        breakdown: Dict[str, float] = {}
        causes_found = 0

        # Root causes: dynamic services from the selected cascade variant
        cause_scores: List[float] = []
        for idx, service in enumerate(cause_services, start=1):
            service_terms = [service]
            if idx == 1:
                service_terms.extend(["embedding service", "embedding"])
            elif idx == 2:
                service_terms.extend(["feature store", "featurestore"])

            cause_score = 0.0
            if contains_any(text, service_terms):
                cause_score = 0.20
                causes_found += 1
            cause_scores.append(cause_score)
            breakdown[f"identifies_{breakdown_label(service)}_issue"] = cause_score

        # Backwards-compatible aliases for callers that inspect the old keys.
        emb_score = cause_scores[0]
        fs_score = cause_scores[1]
        third_score = cause_scores[2]
        breakdown["identifies_embedding_service_issue"] = emb_score
        breakdown["identifies_feature_store_cache_issue"] = fs_score
        third_is_ab_router = contains_term(normalize_text(cause_services[2]), "ab_test_router")
        if third_is_ab_router:
            breakdown["identifies_ab_router_misconfiguration"] = third_score
        else:
            breakdown["identifies_third_service_issue"] = third_score

        score += sum(cause_scores)

        # Root cause 1 issue evidence
        if emb_score == 0.0 and contains_any(
            text,
            [
                "onnx",
                "runtime mismatch",
                "corrupted embedding",
                "cuda",
                "gpu unavailable",
                "dimension changed",
                "downstream consumers incompatible",
            ],
        ):
            emb_score = 0.20
            causes_found += 1
            breakdown[f"identifies_{breakdown_label(cause_services[0])}_issue"] = emb_score
            breakdown["identifies_embedding_service_issue"] = emb_score
            score += emb_score

        # Root cause 2 issue evidence
        if fs_score == 0.0 and contains_any(
            text,
            [
                "cache",
                "ttl",
                "stale feature",
                "redis",
                "connection pool",
                "serialized feature reads",
                "schema version mismatch",
            ],
        ):
            fs_score = 0.20
            causes_found += 1
            breakdown[f"identifies_{breakdown_label(cause_services[1])}_issue"] = fs_score
            breakdown["identifies_feature_store_cache_issue"] = fs_score
            score += fs_score

        # Root cause 3 issue evidence
        if third_score == 0.0 and contains_any(
            text,
            [
                "traffic split",
                "model b",
                "routing config",
                "ab test",
                "wrong model artifact",
                "staging model weights",
                "experiment config",
                "control group",
            ],
        ):
            third_score = 0.20
            causes_found += 1
            breakdown[f"identifies_{breakdown_label(cause_services[2])}_issue"] = third_score
            if third_is_ab_router:
                breakdown["identifies_ab_router_misconfiguration"] = third_score
            else:
                breakdown["identifies_third_service_issue"] = third_score
            score += third_score

        # Synthesis: connects all 3 to single deployment
        synthesis_score = 0.0
        if causes_found >= 3 and contains_any(
            text,
            [deployment, "single deployment", "same deploy", "deployment v", "deploy rollout", "one change caused"],
        ):
            synthesis_score = 0.15
        elif causes_found >= 2:
            synthesis_score = 0.05
        breakdown["connects_all_causes_to_deployment"] = synthesis_score
        score += synthesis_score

        # Remediation: coordinated rollback
        rollback_score = 0.0
        if contains_any(text, ["rollback", "revert", "coordinated", "all services", "full rollback", "restore all"]):
            rollback_score = 0.15 if causes_found >= 3 else 0.07
        breakdown["proposes_coordinated_rollback"] = rollback_score
        score += rollback_score

        # Investigation breadth bonus
        inv_breadth = sum(
            any(contains_term(p, service) for p in path)
            for service in cause_services
        )
        inv_bonus = round((inv_breadth / 3) * 0.10, 4)
        breakdown["investigation_breadth_bonus"] = inv_bonus
        score += inv_bonus

        # Cap score based on number of causes found (prevents partial guessing)
        if causes_found == 0:
            score = min(score, 0.10)
        elif causes_found == 1:
            score = min(score, 0.35)
        elif causes_found == 2:
            score = min(score, 0.70)

        breakdown["causes_found"] = float(causes_found)
        return {"total": round(min(max(score, 0.0001), 0.9999), 4), "breakdown": breakdown}
