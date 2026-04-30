from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


def _read_env_token() -> str:
    env_path = Path(".env")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("HF_TOKEN="):
            return s.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN missing in .env")


def main() -> int:
    token = _read_env_token()
    repo_id = "jason9150/mlops-incident-agent-grpo-hf"
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj="traces/oracle.jsonl",
        path_in_repo="traces/oracle.jsonl",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="data/oracle_sft.jsonl",
        path_in_repo="data/oracle_sft.jsonl",
        repo_id=repo_id,
        repo_type="model",
    )
    print("upload_ok", repo_id, "traces/oracle.jsonl", "data/oracle_sft.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
