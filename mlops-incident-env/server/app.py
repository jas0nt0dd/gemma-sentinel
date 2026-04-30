"""
app.py — FastAPI server for MLOps Incident Response Environment.

OpenEnv HTTP protocol:
  POST /reset   → initial observation
  POST /step    → action → observation + reward + done
  GET  /state   → episode metadata (root cause hidden)
  GET  /health  → liveness check
  WS   /ws      → WebSocket (required by openenv Python client)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .environment import MLOpsEnvironment
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from environment import MLOpsEnvironment
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Incident Response Environment",
    description=(
        "OpenEnv-compatible RL environment where AI agents act as on-call ML engineers. "
        "Diagnose and resolve production ML incidents across 3 difficulty levels."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton env (one per worker) ────────────────────────────────────────────
env = MLOpsEnvironment()
reset_response_cache: Dict[str, Dict[str, Any]] = {}
step_response_cache: Dict[str, Dict[str, Any]] = {}


# ── Request schemas ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = Field(
        default="easy",
        description="Task difficulty: 'easy' | 'medium' | 'hard'",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for retry-safe resets.",
    )


class StepRequest(BaseModel):
    action_type: str = Field(
        default="inspect",
        description=(
            "inspect | query_logs | check_metrics | compare_configs | "
            "check_feature_drift | submit_diagnosis | request_rollback"
        ),
    )
    target: str = Field(default="", description="Component name or root cause string")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra params e.g. {'root_cause': '...', 'fix': '...'}",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for retry-safe steps.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    """Human-friendly root endpoint for Spaces/browser health probes."""
    return {
        "name": "MLOps Incident Response Environment",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "training_artifacts": "/training-artifacts",
        "openenv_endpoints": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "ws": "/ws",
        },
    }


@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "mlops-incident-env", "version": "1.0.0"}


@app.get("/training-artifacts", tags=["system"])
def training_artifacts() -> Dict[str, str]:
    """Hub links for GRPO metrics/plot (after a job or local run pushed with HF_TOKEN)."""
    repo = os.getenv("HF_TRAINING_ARTIFACTS_REPO", "jason9150/mlops-incident-agent-grpo-hf").strip()
    base = f"https://huggingface.co/{repo}"
    return {
        "model_repo": base,
        "metrics_json": f"{base}/resolve/main/metrics.json",
        "reward_plot_png": f"{base}/resolve/main/grpo_reward_curves.png",
        "hf_train_py": f"{base}/resolve/main/hf_train.py",
    }


@app.post("/reset", tags=["openenv"])
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """Reset environment. validate-submission.sh pings this — must return 200."""
    task_id = request.task_id if request else "easy"
    request_id = request.request_id if request else None
    if request_id and request_id in reset_response_cache:
        return reset_response_cache[request_id]
    if task_id not in ("easy", "medium", "hard", "cascade"):
        task_id = "easy"
    obs = env.reset(task_id=task_id)
    response = obs.to_dict()
    step_response_cache.clear()
    if request_id:
        reset_response_cache[request_id] = response
    return response


@app.post("/step", tags=["openenv"])
def step(request: StepRequest) -> Dict[str, Any]:
    """Execute one action, return observation + reward + done."""
    if request.request_id and request.request_id in step_response_cache:
        return step_response_cache[request.request_id]
    obs = env.step(
        action_type=request.action_type,
        target=request.target,
        parameters=request.parameters,
    )
    response = obs.to_dict()
    if request.request_id:
        step_response_cache[request.request_id] = response
    return response


@app.get("/state", tags=["openenv"])
def state() -> Dict[str, Any]:
    """Episode metadata. true_root_cause excluded intentionally."""
    return env.state.to_dict()


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Required by openenv Python client (HTTPEnvClient).

    Client sends JSON:
      {"command": "reset", "task_id": "easy"}
      {"command": "step", "action_type": "inspect", "target": "api_gateway", "parameters": {}}
      {"command": "state"}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            command = msg.get("command", "")

            if command == "reset":
                obs = env.reset(task_id=msg.get("task_id", "easy"))
                await websocket.send_json(obs.to_dict())

            elif command == "step":
                obs = env.step(
                    action_type=msg.get("action_type", "inspect"),
                    target=msg.get("target", ""),
                    parameters=msg.get("parameters", {}),
                )
                await websocket.send_json(obs.to_dict())

            elif command == "state":
                await websocket.send_json(env.state.to_dict())

            else:
                await websocket.send_json(
                    {"error": f"Unknown command '{command}'. Use: reset, step, state"}
                )

    except WebSocketDisconnect:
        pass

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")

if __name__ == "__main__":
    main()

