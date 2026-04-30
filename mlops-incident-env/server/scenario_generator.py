"""
scenario_generator.py — Dynamic Scenario Generator for MLOps Incident Response Env
Generates randomized-but-realistic incidents every episode.
Static JSONs are now fallback only. This is the primary scenario source.
"""
from __future__ import annotations
import random
from typing import Any, Dict


class ScenarioGenerator:
    """Generates a fresh scenario dict every call. Same structure as static JSONs."""

    # ── Easy: Data Quality / Schema Break ─────────────────────────────────────
    _EASY_VARIANTS = [
        {
            "broken_pipeline": "data_pipeline_b",
            "healthy_pipeline": "data_pipeline_a",
            "field": "transaction_amount",
            "migration": "v2.3.1",
            "from_type": "FLOAT",
            "to_type": "STRING",
            "null_rate": 0.34,
            "accuracy_drop": 18,
            "accuracy_current": 0.69,
            "accuracy_sla": 0.80,
            "red_herring": "api_gateway",
        },
        {
            "broken_pipeline": "data_pipeline_a",
            "healthy_pipeline": "data_pipeline_b",
            "field": "user_session_duration",
            "migration": "v3.1.0",
            "from_type": "INTEGER",
            "to_type": "FLOAT",
            "null_rate": 0.28,
            "accuracy_drop": 14,
            "accuracy_current": 0.72,
            "accuracy_sla": 0.82,
            "red_herring": "model_server",
        },
        {
            "broken_pipeline": "data_pipeline_c",
            "healthy_pipeline": "data_pipeline_a",
            "field": "purchase_count_7d",
            "migration": "v1.9.4",
            "from_type": "INT",
            "to_type": "STRING",
            "null_rate": 0.41,
            "accuracy_drop": 22,
            "accuracy_current": 0.65,
            "accuracy_sla": 0.80,
            "red_herring": "monitoring_service",
        },
    ]

    # ── Medium: Latency Spike / Config Change ──────────────────────────────────
    _MEDIUM_VARIANTS = [
        {
            "broken_service": "feature_preprocessor_v2",
            "config_param": "batch_size",
            "old_value": 32,
            "new_value": 512,
            "symptom": "OOM / memory leak",
            "latency_before": 45,
            "latency_after": 847,
            "timeout_rate": 0.23,
            "red_herring": "cache_service",
        },
        {
            "broken_service": "feature_preprocessor_v2",
            "config_param": "worker_threads",
            "old_value": 4,
            "new_value": 64,
            "symptom": "thread contention / CPU saturation",
            "latency_before": 52,
            "latency_after": 1100,
            "timeout_rate": 0.31,
            "red_herring": "api_gateway",
        },
        {
            "broken_service": "model_serving",
            "config_param": "max_concurrent_requests",
            "old_value": 100,
            "new_value": 8,
            "symptom": "request queue overflow",
            "latency_before": 38,
            "latency_after": 920,
            "timeout_rate": 0.27,
            "red_herring": "load_balancer",
        },
    ]

    # ── Hard: Silent Feature Drift ─────────────────────────────────────────────
    _HARD_VARIANTS = [
        {
            "drifted_feature": "user_engagement_score",
            "psi": 0.38,
            "experiment": "A441",
            "experiment_desc": "UI redesign — new home feed layout",
            "model_age_days": 60,
            "revenue_drop_pct": 12.3,
            "training_mean": 0.72,
            "current_mean": 0.51,
            "shap_importance": 0.34,
        },
        {
            "drifted_feature": "purchase_frequency_30d",
            "psi": 0.44,
            "experiment": "B209",
            "experiment_desc": "Checkout flow redesign — removed guest checkout",
            "model_age_days": 45,
            "revenue_drop_pct": 9.8,
            "training_mean": 3.2,
            "current_mean": 1.7,
            "shap_importance": 0.29,
        },
        {
            "drifted_feature": "avg_order_value",
            "psi": 0.31,
            "experiment": "C117",
            "experiment_desc": "Pricing experiment — dynamic surge pricing enabled",
            "model_age_days": 75,
            "revenue_drop_pct": 15.1,
            "training_mean": 48.5,
            "current_mean": 71.2,
            "shap_importance": 0.26,
        },
    ]

    # ── Cascade: Multi-Root Deployment Failure ─────────────────────────────────
    _CASCADE_VARIANTS = [
        {
            "deployment": "v7.8.2",
            "cause1_service": "embedding_service_v3",
            "cause1_issue": "ONNX runtime version mismatch (expected 1.15, found 1.18)",
            "cause2_service": "feature_store",
            "cause2_issue": "cache TTL set to 0s — all features stale instantly",
            "cause3_service": "ab_test_router",
            "cause3_issue": "traffic split corrupted — 100% routed to untested model_B",
            "revenue_impact": "50K/hr",
        },
        {
            "deployment": "v8.1.0",
            "cause1_service": "embedding_service_v3",
            "cause1_issue": "CUDA driver version downgraded — GPU unavailable, CPU fallback 12x slower",
            "cause2_service": "feature_store",
            "cause2_issue": "Redis connection pool size set to 1 — serialized all feature reads",
            "cause3_service": "model_registry",
            "cause3_issue": "wrong model artifact path — production pointing to staging model weights",
            "revenue_impact": "35K/hr",
        },
        {
            "deployment": "v9.0.1",
            "cause1_service": "embedding_service_v3",
            "cause1_issue": "embedding dimension changed 128→256 — downstream consumers incompatible",
            "cause2_service": "feature_store",
            "cause2_issue": "feature schema version mismatch — v2 features fed to v3 model",
            "cause3_service": "ab_test_router",
            "cause3_issue": "experiment config overwritten — control group eliminated, 100% in treatment",
            "revenue_impact": "65K/hr",
        },
    ]

    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, task_id: str, seed: int | None = None) -> Dict[str, Any]:
        """Return a fresh scenario dict matching the static JSON structure."""
        rng = random.Random(seed)
        if task_id == "easy":
            return self._build_easy(rng.choice(self._EASY_VARIANTS), rng)
        elif task_id == "medium":
            return self._build_medium(rng.choice(self._MEDIUM_VARIANTS), rng)
        elif task_id == "hard":
            return self._build_hard(rng.choice(self._HARD_VARIANTS), rng)
        elif task_id == "cascade":
            return self._build_cascade(rng.choice(self._CASCADE_VARIANTS), rng)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    # ── Builders ───────────────────────────────────────────────────────────────

    def _build_easy(self, v: Dict, rng: random.Random) -> Dict[str, Any]:
        bp = v["broken_pipeline"]
        hp = v["healthy_pipeline"]
        field = v["field"]
        mig = v["migration"]
        null_rate = v["null_rate"]
        acc = v["accuracy_current"]
        drop = v["accuracy_drop"]
        rh = v["red_herring"]
        t = rng.randint(15, 25)

        return {
            "task_id": "easy",
            "scenario_name": "Data Quality Alert",
            "goal": (
                f"Model accuracy dropped {drop}% in the last 2 hours. SLA breach in 30 minutes. "
                f"Investigate and identify the root cause. "
                f"Use: inspect, query_logs, check_metrics, submit_diagnosis."
            ),
            "alert_summary": (
                f"[CRITICAL] Recommendation model accuracy: {acc:.2f} (SLA: {v['accuracy_sla']:.2f}). "
                f"Breach in 30 min. Triggered at 14:{t:02d} UTC."
            ),
            "max_steps": 10,
            "sla_minutes": 30,
            "sla_steps": 8,
            "components": {
                rh: {
                    "status": "healthy",
                    "description": "Handles all incoming inference requests. Routes to model_server.",
                    "last_updated": f"14:{t-2:02d} UTC",
                    "logs": [
                        {"time": f"14:{t-2:02d} UTC", "level": "INFO", "msg": "Request throughput normal: 1,240 req/s"},
                        {"time": f"14:{t:02d} UTC", "level": "INFO", "msg": "All upstream routes healthy"},
                    ],
                    "metrics": {"request_rate": 1240, "error_rate": 0.001, "p99_latency_ms": 143},
                },
                "feature_store": {
                    "status": "degraded",
                    "description": f"Pre-computes and caches model features. Fed by {hp} and {bp}.",
                    "last_updated": f"14:{t:02d} UTC",
                    "logs": [
                        {"time": f"14:{t-12:02d} UTC", "level": "WARN",
                         "msg": f"Received {int(null_rate*1000)} null values in batch. Field: {field}."},
                        {"time": f"14:{t-6:02d} UTC", "level": "WARN",
                         "msg": f"Null rate rising: {int(null_rate*35)}% of records missing {field}"},
                        {"time": f"14:{t-1:02d} UTC", "level": "ERROR",
                         "msg": f"Null rate critical: {int(null_rate*100)}% of records missing {field}. Models may degrade."},
                    ],
                    "metrics": {
                        "null_feature_rate": null_rate,
                        "batch_success_rate": round(1 - null_rate, 2),
                        "cache_hit_rate": 0.91,
                    },
                },
                hp: {
                    "status": "healthy",
                    "description": "Ingests upstream data. Feeds feature_store.",
                    "last_updated": f"14:{t:02d} UTC",
                    "logs": [
                        {"time": f"14:{t:02d} UTC", "level": "INFO", "msg": "Batch processed successfully. 0 errors."},
                        {"time": f"14:{t:02d} UTC", "level": "INFO", "msg": "Schema validation passed. All fields nominal."},
                    ],
                    "metrics": {"records_processed": 98200, "error_rate": 0.0, "schema_violations": 0},
                },
                bp: {
                    "status": "error",
                    "description": (
                        f"Ingests transaction data. Feeds feature_store. "
                        f"Schema migration {mig} applied at 14:{t-15:02d} UTC."
                    ),
                    "last_updated": f"14:{t-15:02d} UTC",
                    "logs": [
                        {"time": f"14:{t-15:02d} UTC", "level": "INFO",
                         "msg": f"Schema migration {mig} applied. Field '{field}' type changed: {v['from_type']} -> {v['to_type']}."},
                        {"time": f"14:{t-14:02d} UTC", "level": "ERROR",
                         "msg": f"Schema validation FAILED: feature_store expects {v['from_type']} for '{field}', received {v['to_type']}. Batch rejected."},
                        {"time": f"14:{t-12:02d} UTC", "level": "ERROR",
                         "msg": "Schema validation FAILED: 1,020 records dropped."},
                        {"time": f"14:{t-1:02d} UTC", "level": "CRITICAL",
                         "msg": f"Cumulative data loss: 3,420 records with corrupted {field}. Notify on-call."},
                    ],
                    "metrics": {
                        "records_processed": 4100,
                        "error_rate": 0.41,
                        "schema_violations": 3420,
                        "last_successful_batch": f"14:{t-16:02d} UTC",
                    },
                },
                "model_server": {
                    "status": "degraded",
                    "description": "Serves the recommendation model. Receives features from feature_store.",
                    "last_updated": f"14:{t:02d} UTC",
                    "logs": [
                        {"time": f"14:{t-8:02d} UTC", "level": "WARN",
                         "msg": f"Feature vector incomplete for {int(null_rate*35)}% of requests. Using default 0.0 for {field}."},
                        {"time": f"14:{t:02d} UTC", "level": "CRITICAL",
                         "msg": f"Accuracy SLA breach: current accuracy {acc:.2f}, threshold {v['accuracy_sla']:.2f}."},
                    ],
                    "metrics": {
                        "accuracy": acc,
                        "accuracy_1h_ago": round(acc + 0.18, 2),
                        "accuracy_sla": v["accuracy_sla"],
                        "incomplete_feature_rate": null_rate,
                    },
                },
                "monitoring_service": {
                    "status": "healthy",
                    "description": "Tracks system health and fires alerts.",
                    "last_updated": f"14:{t:02d} UTC",
                    "logs": [
                        {"time": f"14:{t:02d} UTC", "level": "CRITICAL",
                         "msg": f"ALERT FIRED: model_accuracy below SLA. Current: {acc:.2f}."},
                    ],
                    "metrics": {"alerts_fired_last_1h": 1, "services_degraded": 1, "services_error": 1},
                },
            },
            "global_metrics": {
                "model_accuracy": {"current": acc, "1h_ago": round(acc + 0.18, 2), "sla": v["accuracy_sla"]},
                "overall_null_rate": {"current": null_rate, "1h_ago": 0.02},
                "system_request_rate": 1240,
            },
            "root_cause": f"{bp}_schema_mismatch",
            "root_cause_keywords": [bp, "schema", field, "migration"],
            "ground_truth": {
                "primary_component": bp,
                "related_components": ["feature_store", "model_server"],
                "red_herrings": [rh],
                "incident_pattern": "schema_drift",
            },
            "reward_config": {
                "relevant_components": [bp, "feature_store", "model_server"],
                "key_evidence_logs": [bp, "feature_store"],
                "explore_relevant_reward": 0.05,
                "find_key_evidence_reward": 0.10,
                "correct_diagnosis_reward": 0.50,
                "partial_diagnosis_reward": 0.25,
                "loop_penalty": -0.15,
                "false_diagnosis_penalty": -0.10,
                "step_over_max_penalty": -0.05,
            },
        }

    def _build_medium(self, v: Dict, rng: random.Random) -> Dict[str, Any]:
        svc = v["broken_service"]
        param = v["config_param"]
        old_val = v["old_value"]
        new_val = v["new_value"]
        lat_after = v["latency_after"]
        timeout = v["timeout_rate"]
        rh = v["red_herring"]
        dh = rng.randint(8, 11)

        return {
            "task_id": "medium",
            "scenario_name": "Latency Spike Investigation",
            "goal": (
                f"P99 inference latency spiked from {v['latency_before']}ms to {lat_after}ms. "
                f"{int(timeout*100)}% of requests are timing out. SLA is 200ms. "
                f"A deployment was made today. Identify the bottleneck component AND the specific "
                f"config change responsible. Use: inspect, query_logs, check_metrics, compare_configs, submit_diagnosis."
            ),
            "alert_summary": (
                f"[CRITICAL] P99 latency: {lat_after}ms (SLA: 200ms). "
                f"{int(timeout*100)}% request timeout. Deployment at {dh:02d}:15 UTC."
            ),
            "max_steps": 15,
            "sla_minutes": 20,
            "sla_steps": 12,
            "components": {
                rh: {
                    "status": "healthy",
                    "description": "Entry point. Not involved in this incident.",
                    "last_updated": "12:44 UTC",
                    "logs": [
                        {"time": "12:44 UTC", "level": "INFO",
                         "msg": f"Upstream timeout rate: {int(timeout*100)}%. Issue is latency, not volume."},
                    ],
                    "metrics": {"request_rate": 890, "error_rate": timeout, "p99_latency_ms": lat_after},
                },
                "load_balancer": {
                    "status": "healthy",
                    "description": f"Distributes load. {svc} was deployed at {dh:02d}:15 UTC.",
                    "last_updated": "12:44 UTC",
                    "logs": [
                        {"time": "12:30 UTC", "level": "WARN",
                         "msg": f"{svc} response time degrading: avg {int(lat_after*0.73)}ms."},
                        {"time": "12:44 UTC", "level": "ERROR",
                         "msg": f"{svc} avg response time: {int(lat_after*0.92)}ms. Health check failing."},
                    ],
                    "metrics": {"v2_avg_latency_ms": int(lat_after * 0.92), "v1_avg_latency_ms": 41},
                },
                svc: {
                    "status": "degraded",
                    "description": f"Deployed {dh:02d}:15 UTC. Processes features before model inference.",
                    "last_updated": f"{dh:02d}:15 UTC",
                    "logs": [
                        {"time": f"{dh:02d}:15 UTC", "level": "INFO",
                         "msg": f"Deployment complete. config: {param}={new_val}."},
                        {"time": "10:15 UTC", "level": "WARN",
                         "msg": f"Processing time increasing. {v['symptom']} detected."},
                        {"time": "12:44 UTC", "level": "CRITICAL",
                         "msg": f"Performance critical. {v['symptom']}. Recommend immediate rollback."},
                    ],
                    "metrics": {
                        "avg_latency_ms": lat_after,
                        param: new_val,
                        "error_rate": timeout,
                        "issue": v["symptom"],
                    },
                },
                "model_server": {
                    "status": "degraded",
                    "description": f"Waits for features from {svc}. Blocking on slow responses.",
                    "last_updated": "12:44 UTC",
                    "logs": [
                        {"time": "12:44 UTC", "level": "ERROR",
                         "msg": f"{int(timeout*100)}% of requests timed out waiting for {svc}."},
                    ],
                    "metrics": {"upstream_timeout_rate": timeout, "p99_latency_ms": lat_after},
                },
            },
            "config_diff": {
                "service": svc,
                "parameter": param,
                "old_value": old_val,
                "new_value": new_val,
                "changed_by": "infra-bot",
                "changed_at": f"{dh:02d}:15 UTC",
                "diff_summary": f"{param}: {old_val} → {new_val}",
            },
            "global_metrics": {
                "p99_latency_ms": {"current": lat_after, "1h_ago": v["latency_before"], "sla": 200},
                "timeout_rate": {"current": timeout, "1h_ago": 0.001},
            },
            "root_cause": f"{svc}_{param}_misconfigured",
            "root_cause_keywords": [svc, param, str(new_val), "rollback"],
            "ground_truth": {
                "primary_component": svc,
                "related_components": ["model_server", "load_balancer"],
                "red_herrings": [rh],
                "incident_pattern": "config_regression",
            },
            "reward_config": {
                "relevant_components": [svc, "load_balancer", "model_server"],
                "key_evidence_logs": [svc, "model_server", "load_balancer"],
                "explore_relevant_reward": 0.05,
                "find_key_evidence_reward": 0.10,
                "correct_diagnosis_reward": 0.50,
                "partial_diagnosis_reward": 0.20,
                "loop_penalty": -0.15,
                "false_diagnosis_penalty": -0.10,
                "step_over_max_penalty": -0.05,
            },
        }

    def _build_hard(self, v: Dict, rng: random.Random) -> Dict[str, Any]:
        feat = v["drifted_feature"]
        psi = v["psi"]
        exp = v["experiment"]
        exp_desc = v["experiment_desc"]
        age = v["model_age_days"]
        rev_drop = v["revenue_drop_pct"]

        stable_features = {
            "user_age_bucket": {"psi": round(rng.uniform(0.01, 0.03), 2), "status": "stable", "note": "No change."},
            "purchase_frequency_30d": {"psi": round(rng.uniform(0.02, 0.05), 2), "status": "stable", "note": "Normal seasonal variation."},
            "avg_order_value": {"psi": round(rng.uniform(0.01, 0.04), 2), "status": "stable", "note": "No change."},
            "device_type": {"psi": round(rng.uniform(0.01, 0.02), 2), "status": "stable", "note": "No change."},
            "geo_region": {"psi": round(rng.uniform(0.02, 0.04), 2), "status": "stable", "note": "No change."},
            "category_affinity_score": {"psi": round(rng.uniform(0.07, 0.12), 2), "status": "minor_drift", "note": "Minor drift, within acceptable range."},
            "recency_days_since_last_purchase": {"psi": round(rng.uniform(0.04, 0.08), 2), "status": "stable", "note": "No change."},
        }
        # Remove the drifted feature from stable list if present, then add it as CRITICAL
        stable_features.pop(feat, None)
        stable_features[feat] = {
            "psi": psi,
            "status": "CRITICAL_DRIFT",
            "note": (
                f"Training distribution: mean={v['training_mean']}. Current: mean={v['current_mean']}. "
                f"Shifted 3 days ago — coincides with experiment #{exp} launch ({exp_desc}). "
                f"#1 feature by SHAP importance (weight: {v['shap_importance']})."
            ),
            "training_stats": {"mean": v["training_mean"]},
            "current_stats": {"mean": v["current_mean"]},
        }

        return {
            "task_id": "hard",
            "scenario_name": "Silent Model Drift",
            "goal": (
                f"Revenue has dropped {rev_drop}% over the past 3 days. "
                f"No error alerts. No system failures. All services appear healthy. "
                f"Investigate the ML system for silent degradation. "
                f"Use: inspect, query_logs, check_metrics, check_feature_drift, submit_diagnosis."
            ),
            "alert_summary": (
                f"[BUSINESS] Revenue anomaly. 3-day revenue drop: -{rev_drop}%. "
                f"No engineering alerts. Escalated manually to ML on-call."
            ),
            "max_steps": 25,
            "components": {
                "api_gateway": {
                    "status": "healthy",
                    "description": "All routes healthy. Request volume stable.",
                    "last_updated": "10:00 UTC",
                    "logs": [{"time": "10:00 UTC", "level": "INFO", "msg": "All systems nominal. Request rate: 2,100 req/s."}],
                    "metrics": {"request_rate": 2100, "error_rate": 0.0, "p99_latency_ms": 62},
                },
                "model_server": {
                    "status": "healthy",
                    "description": f"Serves recommendation model v4.2. Trained {age} days ago. No errors.",
                    "last_updated": "10:00 UTC",
                    "logs": [
                        {"time": "10:00 UTC", "level": "INFO", "msg": "Model v4.2 serving normally. 0 errors."},
                        {"time": "09:50 UTC", "level": "INFO", "msg": f"Model last retrained: {age} days ago. Retraining pipeline: IDLE."},
                    ],
                    "metrics": {"model_version": "v4.2", "model_trained_days_ago": age, "error_rate": 0.0},
                },
                "feature_store": {
                    "status": "healthy",
                    "description": "Pre-computes all model features. All pipelines normal.",
                    "last_updated": "10:00 UTC",
                    "logs": [{"time": "10:00 UTC", "level": "INFO", "msg": "All feature pipelines running. No errors."}],
                    "metrics": {"null_rate": 0.001, "schema_violations": 0, "batch_success_rate": 1.0},
                },
                "ab_testing_service": {
                    "status": "healthy",
                    "description": f"Runs A/B experiments. Experiment #{exp} launched 3 days ago: {exp_desc}.",
                    "last_updated": "10:00 UTC",
                    "logs": [
                        {"time": "Day -3, 00:00 UTC", "level": "INFO",
                         "msg": f"Experiment #{exp} launched: {exp_desc}. Rolled out to 100% of users."},
                        {"time": "Day -2, 10:00 UTC", "level": "INFO",
                         "msg": f"Experiment #{exp}: User engagement patterns shifting significantly."},
                    ],
                    "metrics": {"experiment_status": "running", "experiment_start": "3 days ago"},
                },
                "monitoring_service": {
                    "status": "healthy",
                    "description": "Monitors error rates and latency. Does NOT track feature drift. No alerts fired.",
                    "last_updated": "10:00 UTC",
                    "logs": [
                        {"time": "10:00 UTC", "level": "INFO", "msg": "All monitored metrics within normal bounds. No alerts."},
                        {"time": "10:00 UTC", "level": "INFO",
                         "msg": "NOTE: Feature drift monitoring disabled — config flag 'enable_psi_alerting=false'."},
                    ],
                    "metrics": {"alerts_fired_last_7d": 0, "psi_alerting_enabled": False},
                },
            },
            "feature_drift_data": {
                "summary": "PSI scores for all model features. PSI > 0.20 = significant. PSI > 0.25 = critical.",
                "features": stable_features,
                "model_feature_importance": {feat: v["shap_importance"]},
            },
            "business_metrics": {
                "revenue": {"pct_change_3d": -rev_drop},
            },
            "root_cause": f"{feat}_drift_model_staleness_{age}_days",
            "root_cause_keywords": [feat, "drift", "psi", "retrain", exp.lower(), f"{age}_days"],
            "ground_truth": {
                "primary_component": "model_server",
                "related_components": ["ab_testing_service", "feature_store", "monitoring_service"],
                "red_herrings": ["api_gateway"],
                "incident_pattern": "feature_drift",
            },
            "reward_config": {
                "relevant_components": ["model_server", "ab_testing_service", "monitoring_service"],
                "key_evidence_logs": ["model_server", "ab_testing_service", "feature_store", "monitoring_service"],
                "key_evidence_drift": [feat],
                "explore_relevant_reward": 0.04,
                "find_feature_drift_reward": 0.15,
                "find_critical_feature_reward": 0.20,
                "correct_diagnosis_reward": 0.40,
                "partial_diagnosis_reward": 0.20,
                "loop_penalty": -0.10,
                "false_diagnosis_penalty": -0.08,
                "step_over_max_penalty": -0.05,
            },
        }

    def _build_cascade(self, v: Dict, rng: random.Random) -> Dict[str, Any]:
        deploy = v["deployment"]
        c1 = v["cause1_service"]
        c2 = v["cause2_service"]
        c3 = v["cause3_service"]

        return {
            "task_id": "cascade",
            "scenario_name": "Cascade Failure (Elite)",
            "goal": (
                f"Three ML services degraded simultaneously 2 hours after deployment {deploy}. "
                f"Model accuracy -22%, revenue impact {v['revenue_impact']}. "
                f"Find ALL root causes and propose a coordinated fix."
            ),
            "alert_summary": (
                f"[CRITICAL] Multi-system failure post deployment {deploy}. "
                f"{c1}, {c2}, {c3} all degraded. Revenue impact: {v['revenue_impact']}."
            ),
            "max_steps": 30,
            "sla_minutes": 20,
            "sla_steps": 15,
            "global_metrics": {
                "model_accuracy_delta": "-22%",
                "revenue_impact": f"-{v['revenue_impact']}",
                "services_degraded": 3,
                "deployment": deploy,
            },
            "components": {
                c1: {
                    "status": "CRITICAL",
                    "description": v["cause1_issue"],
                    "last_updated": "14:02 UTC",
                    "logs": [
                        {"time": "14:05 UTC", "level": "CRITICAL", "msg": v["cause1_issue"]},
                    ],
                    "metrics": {"error_rate": "67%", "status": "critical"},
                },
                c2: {
                    "status": "CRITICAL",
                    "description": v["cause2_issue"],
                    "last_updated": "14:02 UTC",
                    "logs": [
                        {"time": "14:03 UTC", "level": "CRITICAL", "msg": v["cause2_issue"]},
                    ],
                    "metrics": {"status": "critical"},
                },
                c3: {
                    "status": "DEGRADED",
                    "description": v["cause3_issue"],
                    "last_updated": "14:01 UTC",
                    "logs": [
                        {"time": "14:02 UTC", "level": "WARNING", "msg": v["cause3_issue"]},
                    ],
                    "metrics": {"status": "degraded"},
                },
                "model_serving": {
                    "status": "degraded",
                    "description": "Downstream of all 3 failed services. Accuracy degraded.",
                    "last_updated": "14:10 UTC",
                    "logs": [
                        {"time": "14:10 UTC", "level": "ERROR",
                         "msg": "Model accuracy: 0.61 (was 0.83). Downstream of multiple failures."},
                    ],
                    "metrics": {"accuracy": 0.61, "accuracy_before": 0.83},
                },
            },
            "deployment_history": [
                {"time": "12:00 UTC", "version": deploy, "author": "ci-bot",
                 "change": "Multi-service config update"},
            ],
            "root_cause": f"cascade_failure_{deploy}",
            "root_cause_keywords": [
                c1.replace("_", " "), c2.replace("_", " "), c3.replace("_", " "),
                "rollback", deploy,
            ],
            "ground_truth": {
                "primary_component": c1,
                "related_components": [c2, c3],
                "red_herrings": ["model_serving"],
                "incident_pattern": "cascade_multi",
            },
            "reward_config": {
                "relevant_components": [c1, c2, c3],
                "key_evidence_logs": [c1, c2, c3],
                "explore_relevant_reward": 0.04,
                "correct_diagnosis_reward": 0.50,
                "partial_diagnosis_reward": 0.15,
                "loop_penalty": -0.12,
                "false_diagnosis_penalty": -0.08,
                "step_over_max_penalty": -0.05,
            },
        }
