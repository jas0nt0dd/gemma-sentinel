---
title: MLOps Incident Response Environment
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# MLOps Incident Response Environment

A production-grade **OpenEnv** environment where LLM agents act as on-call ML engineers: diagnose production incidents from alerts, component status, logs, metrics, drift checks, and config diffs—then submit a structured diagnosis. Built for the **META × Hugging Face OpenEnv Hackathon**.

## Links for judges (submission)

| Item | URL |
|------|-----|
| **Live environment (HF Space)** | https://jason9150-mlops-incident-env.hf.space |
| **Health check** | https://jason9150-mlops-incident-env.hf.space/health |
| **OpenEnv spec (`openenv.yaml`)** | This repo, root `openenv.yaml` |
| **GRPO training script + Hub push** | Model repo: https://huggingface.co/jason9150/mlops-incident-agent-grpo-hf (`hf_train.py`, `metrics.json`, `grpo_reward_curves.png` after a run) |
| **Google Colab (training / repro)** | [Notebook](https://colab.research.google.com/drive/1BAduA4fXaV6QGvjUWi1VZPdyW1-PdTRr?authuser=1#) — end-to-end **Unsloth + TRL GRPO** against the live Space (`HF_SPACE_URL`): install deps, optional oracle SFT, train, eval, plots. Primary artifact for hackathon “training in Colab” expectations. |
| **Hugging Face GPU Job (hosted run)** | [Job `69edb45e…`](https://huggingface.co/jobs/jason9150/69edb45ed70108f37acdfbfa) — **HF Jobs** run of `hf_train.py` on HF-managed GPU (logs, hardware, and timeline on the Job page). Complements Colab: same script path, different compute surface. |
| **Mini-blog (HF Space)** | Draft: [blog.md](https://github.com/jas0nt0dd/mlops-incident-env/blob/main/blog.md) — **publish** on [Space → New Discussion](https://huggingface.co/spaces/jason9150/mlops-incident-env/discussions/new) (not the model repo). Paste the **Discussion URL** in your submission README row. |

---

## Storytelling (30%) — what we built

1. **Problem:** MLOps incidents (data quality, misconfig, silent drift, cascade deploys) need agents that read evidence and reason before acting—not template playbooks.  
2. **Environment:** Four tasks (`easy`, `medium`, `hard`, `cascade`) with process-aware scoring (investigation quality, efficiency, multi-root synthesis). Rewards are dense during the episode, not only 0/1 at the end.  
3. **Training:** **Unsloth + HF TRL GRPO** against the **live Space** (`HF_SPACE_URL`): prompts are built from real `reset`/`step` observations; rewards use env `submit_diagnosis` scores plus small capped format shaping (`hf_train.py`). Optional oracle SFT warm-up via `SFT_ORACLE_JSONL` + `scripts/oracle_traces_to_sft_jsonl.py`.  
4. **Evidence of improvement (20%):** After training, `hf_train.py` writes **`metrics.json`** and **`grpo_reward_curves.png`** (baseline vs post-train, mid-train eval curve) and can push them to the model Hub repo above—**link or paste key deltas in your blog**.

---

## Minimum requirements checklist

- [x] **OpenEnv** — `openenv` on PyPI; spec in `openenv.yaml`.  
- [x] **HF Space** — Dockerized Space (see `Dockerfile`); URL above.  
- [x] **Training script** — `hf_train.py` (Unsloth + TRL `GRPOTrainer`); upload with `python scripts/_submit_hf_job_once.py --upload-only` (needs `HF_TOKEN` + `HF_REPO_ID` in `.env`).  
- [ ] **Mini-blog HF or YouTube &lt;2 min** — **You must publish and paste the link** in the table above and in your final submission form.  
- [x] **Reward / pipeline coherence (10%)** — Env score is teacher reward; shaping is capped; curriculum via `RUN_MODE` / `TRAIN_COUNTS_JSON`.

---

## Judging criteria (quick reference)

| Weight | Criterion |
|--------|-----------|
| 40% | Environment innovation |
| 30% | Storytelling & presentation |
| 20% | Observable training improvement (curves / before–after) |
| 10% | Reward & training pipeline |

---

## Baseline scores (representative agent)

| Task    | Score  | Difficulty |
|---------|--------|------------|
| easy    | 0.9500 | Easy       |
| medium  | 1.0000 | Medium     |
| hard    | 0.9000 | Hard       |
| cascade | 1.0000 | Hard+      |
| **AVERAGE** | **0.9625** | |

*(Update this table if your baseline script produces different numbers.)*

---

## API (for manual / agent testing)

- `POST /reset` — Start episode: `{"task_id": "easy"|"medium"|"hard"|"cascade"}`  
- `POST /step` — Act: `{"action_type": "<action>", "target": "<component>", "parameters": {}}`  
- `GET /health` — Liveness for judges and for `hf_train.py` before training.

Example step:

```json
{"action_type": "inspect", "target": "data_pipeline_a", "parameters": {}}
```

---

## Tasks

| Task    | Scenario                                        | Key Skill              |
|---------|-------------------------------------------------|------------------------|
| Easy    | Data pipeline schema mismatch → accuracy drop | Log analysis           |
| Medium  | Batch size config change → latency spike      | Config comparison      |
| Hard    | Feature drift → silent revenue degradation    | Drift detection        |
| Cascade | Multiple failures from one bad deploy         | Cross-system synthesis |

---

## Local validation (before you submit)

```bash
# OpenEnv spec (passes in CI / dev — run from repo root)
openenv validate

# Docker image (requires Docker Desktop / daemon running)
docker build -t mlops-incident-env:judge .
docker run --rm -p 8000:8000 mlops-incident-env:judge
# Then: curl http://localhost:8000/health
```

Oracle traces → SFT JSONL for warm-up:

```bash
python scripts/oracle_traces_to_sft_jsonl.py traces/oracle.jsonl data/oracle_sft.jsonl
# Set SFT_ORACLE_JSONL to that file path when running hf_train.py
```

---

## Final HF Job (A100 + oracle SFT + mentor-visible artifacts)

1. **Upload traces to the model repo** (so the job sees `data/oracle_sft.jsonl` on the volume):

   ```bash
   python scripts/upload_oracle_artifacts.py
   ```

2. **Push latest `hf_train.py`**, then **submit** (reads `hf_job_final.env` — `HF_JOB_FLAVOR=a100-large`, `SFT_ORACLE_JSONL=/tmp/hf_repo/data/oracle_sft.jsonl`, etc.). The job **downloads** Hub files into **`/tmp/hf_repo`** (writable; `/repo` is not writable in HF Jobs).

   ```bash
   python scripts/_submit_hf_job_once.py --upload-only
   python scripts/_submit_hf_job_once.py --job-only
   ```

   Job stdout (URL / id) is also written to `hf_job_last_submit.txt`. Paste that **job link** in your blog / submission.

   If the log shows the wrong GPU (e.g. `a10g-large` instead of `a100-large`), your **shell** may have `HF_JOB_FLAVOR` set — it overrides `hf_job_final.env`. Unset it, then run `--job-only` again.

3. **Plots & metrics on Hub:** after the job finishes with `HF_TOKEN`, `hf_train.py` pushes `metrics.json` and `grpo_reward_curves.png` to the model repo.

4. **From the live Space:** open **`/training-artifacts`** on your Space (same host as `/health`) for direct links to those Hub files for judges.

---

## Repo layout (high signal)

| Path | Role |
|------|------|
| `openenv.yaml` | Task IDs, descriptions, scoring rubric |
| `Dockerfile` | Space container (Python 3.11 + `server/`) |
| `server/` | FastAPI app, environment, scenarios, tasks |
| `inference.py` | Baseline / evaluation agent (Space + LLM) |
| `hf_train.py` | GRPO fine-tune vs live Space; metrics + plot + Hub push |
| `scripts/_submit_hf_job_once.py` | Upload `hf_train.py` to Hub and/or submit HF Job |
| `hf_job_final.env` | Last-run HF Job env (A100, SFT path, curriculum, `UNSLOTH_DISABLE_STATISTICS`) |
| `scripts/upload_oracle_artifacts.py` | Push `traces/oracle.jsonl` + `data/oracle_sft.jsonl` to model repo |
| `scripts/oracle_traces_to_sft_jsonl.py` | `oracle_trace_v1` JSONL → SFT JSONL |

---

## License

MIT — see repository files for details.
