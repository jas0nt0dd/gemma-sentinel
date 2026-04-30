"""Upload hf_train.py to the Hub model repo and submit an HF Job.

Scheduling (stuck on "Scheduling") is Hugging Face GPU queue / capacity — not a bug in
hf_train.py (that file only runs after a machine is assigned). a10g-small can sit in
queue when demand is high. Use a cheaper tier to get a slot sooner:

  Add to .env:  HF_JOB_FLAVOR=t4-medium
  or:          HF_JOB_FLAVOR=t4-small
  or:          HF_JOB_FLAVOR=l4x1

Omit HF_JOB_FLAVOR to default to t4-medium (usually schedules faster than a10g-small).

Final run (A100 + oracle SFT + env from hf_job_final.env):
  python scripts/upload_oracle_artifacts.py
  python scripts/_submit_hf_job_once.py --upload-only
  python scripts/_submit_hf_job_once.py --job-only

Push only, then submit (no double upload):
  python scripts/_submit_hf_job_once.py --upload-only
  python scripts/_submit_hf_job_once.py --job-only
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# Cheapest GPU tiers typically leave "Scheduling" sooner than a10g-small.
_DEFAULT_FLAVOR = "t4-medium"


def _read_key_from_file(key: str, path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(f"{key}="):
            return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _read_job_env(key: str) -> str | None:
    """Process env wins, then .env, then hf_job_final.env (last file wins on duplicate keys)."""
    v = os.environ.get(key, "").strip()
    if v:
        return v
    val: str | None = None
    for path in (PROJECT / ".env", PROJECT / "hf_job_final.env"):
        got = _read_key_from_file(key, path)
        if got is not None:
            val = got
    return val


def _read_token() -> str | None:
    """HF_TOKEN only from process env or .env (never commit token into hf_job_final.env)."""
    v = os.environ.get("HF_TOKEN", "").strip()
    if v:
        return v
    return _read_key_from_file("HF_TOKEN", PROJECT / ".env")


def _read_config(key: str, default: str = "") -> str:
    """Delegate to _read_job_env (process env > hf_job_final.env > .env)."""
    v = _read_job_env(key)
    return (v if v is not None else "").strip() or default


def _optional_job_envs() -> dict[str, str]:
    keys = [
        "RUN_MODE",
        "RUN_PROFILE",
        "NUM_EPOCHS",
        "LR",
        "NUM_GEN",
        "MAX_NEW_TOK",
        "MAX_SEQ_LENGTH",
        "EVAL_GREEDY",
        "SAVE_BEST_BY_EVAL",
        "MID_EVAL_N_EPS",
        "N_EVAL_EPS",
        "REWARD_MAX_SHAPING",
        "REWARD_SHAPING_MIN_ENV",
        "TRAIN_COUNTS_JSON",
        "SFT_ORACLE_JSONL",
        "SFT_EPOCHS",
        "SFT_LR",
        "SFT_GRAD_ACCUM",
        "OUTPUT_DIR",
        "UNSLOTH_DISABLE_STATISTICS",
    ]
    out: dict[str, str] = {}
    for key in keys:
        val = _read_config(key, "")
        if val != "":
            out[key] = val
    return out


def _validate_model_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip()
    if not repo_id:
        raise SystemExit("HF_REPO_ID is required and must point to a Hugging Face model repo.")
    lowered = repo_id.lower()
    if "huggingface.co/spaces/" in lowered or lowered.startswith("spaces/"):
        raise SystemExit(f"HF_REPO_ID must be a model repo id, not a Space target: {repo_id!r}")
    if repo_id.startswith("http://") or repo_id.startswith("https://"):
        raise SystemExit(f"HF_REPO_ID must be a repo id like 'owner/name', not a URL: {repo_id!r}")
    if repo_id.count("/") != 1:
        raise SystemExit(f"HF_REPO_ID must look like 'owner/name'; got {repo_id!r}")
    return repo_id


def main() -> int:
    upload_only = "--upload-only" in sys.argv
    job_only = "--job-only" in sys.argv
    if upload_only and job_only:
        print("Use either --upload-only or --job-only, not both.", file=sys.stderr)
        return 2

    token = _read_token()
    if not token:
        print("HF_TOKEN missing in .env or environment", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, create_repo

    repo_id = _validate_model_repo_id(
        _read_config("HF_REPO_ID", "jason9150/mlops-incident-agent-grpo-hf")
    )
    space_url = _read_config("HF_SPACE_URL", "https://jason9150-mlops-incident-env.hf.space")
    if not job_only:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        HfApi(token=token).upload_file(
            path_or_fileobj=str(PROJECT / "hf_train.py"),
            path_in_repo="hf_train.py",
            repo_id=repo_id,
            repo_type="model",
        )
        print("upload_ok", repo_id)

    if upload_only:
        print("Done (--upload-only); no job submitted.")
        return 0

    flavor = _read_config("HF_JOB_FLAVOR", _DEFAULT_FLAVOR)
    print(f"Using HF Job flavor: {flavor} (set HF_JOB_FLAVOR in .env or env to override)")
    if not str(flavor).lower().startswith("cpu"):
        print(
            "NOTE: GPU Jobs need prepaid Hugging Face credits (see https://huggingface.co/pricing ). "
            "Without credits, jobs often stay in SCHEDULING and never start. "
            "CPU jobs (e.g. cpu-basic) can still complete — use Billing → Compute to verify balance."
        )

    def _find_hf_cli() -> str | None:
        w = shutil.which("hf")
        if w:
            return w
        for p in (PROJECT / ".venv" / "Scripts" / "hf.exe", PROJECT / ".venv" / "bin" / "hf"):
            if p.is_file():
                return str(p)
        return None

    hf = _find_hf_cli()
    if not hf:
        print("hf CLI not found. Install huggingface_hub[cli] or add hf to PATH.", file=sys.stderr)
        return 1
    env = {**os.environ, "HUGGINGFACE_HUB_TOKEN": token}
    subprocess.run([hf, "auth", "login", "--token", token], env=env, capture_output=True, text=True)

    # hf jobs run no longer supports -v repo mounts; pull hf_train + oracle SFT from Hub then execute.
    # Use /tmp (writable); /repo at filesystem root is permission-denied in HF Job containers.
    remote_bootstrap = r"""set -e
pip install -q huggingface_hub
python << 'PY'
from huggingface_hub import hf_hub_download
import os
import runpy

root = "/tmp/hf_repo"
repo = os.environ["HF_REPO_ID"]
tok = os.environ.get("HF_TOKEN")
os.makedirs(f"{root}/data", exist_ok=True)
hf_hub_download(
    repo_id=repo,
    filename="hf_train.py",
    repo_type="model",
    local_dir=root,
    token=tok,
)
try:
    hf_hub_download(
        repo_id=repo,
        filename="data/oracle_sft.jsonl",
        repo_type="model",
        local_dir=root,
        token=tok,
    )
except Exception:
    pass
runpy.run_path(f"{root}/hf_train.py", run_name="__main__")
PY
"""

    args = [
        "jobs",
        "run",
        "--detach",
        "--namespace",
        "jason9150",
        "--flavor",
        flavor,
        "--timeout",
        "6h",
        "--secrets",
        "HF_TOKEN",
        "-e",
        f"HF_SPACE_URL={space_url}",
        "-e",
        f"HF_REPO_ID={repo_id}",
        "-e",
        "MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct",
        "-e",
        "AUTO_PIP_INSTALL=1",
        "unsloth/unsloth:latest",
        "bash",
        "-lc",
        remote_bootstrap,
    ]
    optional_envs = _optional_job_envs()
    optional_envs.pop("HF_JOB_FLAVOR", None)
    if optional_envs:
        try:
            insert_at = args.index("unsloth/unsloth:latest")
        except ValueError:
            insert_at = len(args)
        cmd = args[:insert_at]
        for k, v in optional_envs.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.extend(args[insert_at:])
        args = cmd
        print("Extra job envs:", ", ".join(sorted(optional_envs.keys())))
    p = subprocess.run([hf, *args], capture_output=True, text=True, env=env)
    out = PROJECT / "hf_job_last_submit.txt"
    out.write_text(
        f"flavor:{flavor}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}\nreturncode:{p.returncode}\n",
        encoding="utf-8",
    )
    print(p.stdout or p.stderr or f"rc={p.returncode}")
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
