# MLOps Incident Response — OpenEnv Hackathon mini-blog

I built this project for the META x Hugging Face OpenEnv Hackathon to train an LLM on a realistic professional workflow: diagnosing production ML incidents from partial evidence, not from static QA prompts.

This post is written for judges. It summarizes the environment, training pipeline, unique data strategy, and where to validate results quickly.

---

## 1. Problem (why this exists)

Production ML fails in ways that are expensive to rehearse: schema drift, bad deploys, silent feature skew, and multi-service cascades. On-call engineers **read evidence** (logs, metrics, configs, drift) and **name the right component** before fixing anything. We built an OpenEnv environment so an LLM can **practice that loop** with **verifiable scores**—not trivia, not a toy grid world.

This aligns with **Theme #3 — World modeling / professional tasks** in the hackathon brief: tools, partial observability, and multi-step workflows instead of shortcuts.

---

## 2. Environment (what the agent does)

**MLOps Incident Response** is OpenEnv-compliant (`openenv.yaml` in the [GitHub repo](https://github.com/jas0nt0dd/mlops-incident-env)):

- **Four tasks:** `easy`, `medium`, `hard`, `cascade` — from a single bad pipeline to a three-way cascade after a bad release.
- **Actions:** inspect components, query logs, check metrics, compare configs, check feature drift, submit a structured diagnosis.
- **Rewards:** Rubric-style scoring (component ID, root cause depth, investigation quality, synthesis, efficiency)—**dense signal during the episode**, not only a terminal 0/1.

**Live demo:**

- **Space:** https://jason9150-mlops-incident-env.hf.space  
- **Health:** https://jason9150-mlops-incident-env.hf.space/health  
- **Training artifacts (metrics + plot links):** https://jason9150-mlops-incident-env.hf.space/training-artifacts  
- **API:** `POST /reset` with `task_id`, then `POST /step` with `action_type`, `target`, `parameters`.

**Training & compute (primary validation links):**

- **Google Colab** — [training / repro notebook](https://colab.research.google.com/drive/1BAduA4fXaV6QGvjUWi1VZPdyW1-PdTRr?authuser=1#): Unsloth + TRL GRPO against the same live Space endpoint used by the environment.
- **Hugging Face GPU Job** — [hosted job `69edb45e…`](https://huggingface.co/jobs/jason9150/69edb45ed70108f37acdfbfa): cloud execution logs, hardware metadata, and run timeline.

---

## 3. Training pipeline (Unsloth + TRL, against the live Space)

Hackathon **minimum requirements** call for a **training script using Unsloth or HF TRL** (ideally Colab). We use **both**:

- **Base model:** Meta **Llama 3.2 3B Instruct** (gated on Hub; license + token required).
- **Method:** **GRPO** (`trl.GRPOTrainer`) + **Unsloth** 4-bit LoRA for efficiency.
- **Reward:** The **environment** scores `submit_diagnosis` (same Space URL as production). Small **capped** format shaping keeps JSON + thinking tags stable without drowning the env signal.
- **Curriculum:** `RUN_MODE` / `TRAIN_COUNTS_JSON` bias easy/medium so a 3B model gets non-zero reward early (per organizer guidance: *success probability must be > 0 for RL to work*).

### Unique strategy: inference trace distillation (our differentiator)

The key strategy I used was to convert high-quality inference trajectories into training signal before RL:

1. Run the agent with tracing enabled and capture structured trajectories in `traces/oracle.jsonl` (thought process, tool usage, and final diagnosis path).
2. Convert traces to supervised format with `scripts/oracle_traces_to_sft_jsonl.py` to create `data/oracle_sft.jsonl`.
3. Run a short oracle SFT warm-up (`SFT_ORACLE_JSONL`) before GRPO.
4. Continue with GRPO on live environment rewards.

Why this matters: direct RL on hard incident tasks can start with near-zero reward. Trace-to-SFT bootstraps policy quality so GRPO gets non-trivial gradient sooner, especially on structured diagnosis output and multi-step investigation behavior.

**Training code & artifacts (separate Hub model repo — adapters + metrics):**  
https://huggingface.co/jason9150/mlops-incident-agent-grpo-hf  
After a run: `metrics.json`, `grpo_reward_curves.png`, and LoRA weights when `HF_TOKEN` is set.

**Source:** https://github.com/jas0nt0dd/mlops-incident-env  

---

## 4. Evidence of improvement (what judges look for — 20%)

Organizers ask for **observable** training progress: *reward curves, metrics, before/after behavior*.

Our runner (`hf_train.py`) logs per-task baseline vs post-train and saves a before/after bar chart plus a mid-training eval curve.

In repeated runs, the most common pattern is:
- easy/medium improve first,
- hard/cascade improve more slowly on a 3B one-shot policy,
- trace-to-SFT warm-up improves early stability and JSON-format success before RL refinement.

**Plots:** label axes (step vs score), keep baseline and trained on **comparable** eval settings (`EVAL_GREEDY=1` by default).

---

## 5. How this maps to judging (official weights)

From **Themes & Judging Criteria** (OpenEnv India 2026):

| Weight | Criterion | How we address it |
|--------|-----------|-------------------|
| **40%** | Environment innovation | MLOps incident domain; multi-tier scenarios; rubric + process-aware rewards; cascade synthesis. |
| **30%** | Storytelling & presentation | This post + README + live Space; problem → env → train → evidence → why it matters. |
| **20%** | Improvement in rewards | Hub / local `metrics.json` + `grpo_reward_curves.png`; before vs after table. |
| **10%** | Reward & training pipeline | Env score is teacher; shaping capped; GRPO connects to **your** Space, not a static offline-only dataset. |

**Minimum checklist:**

- [x] OpenEnv (latest) — spec + server in repo.  
- [x] Training script — Unsloth + TRL (`hf_train.py`).  
- [x] Evidence of a real train — plots + metrics (link files in this Discussion or in README).  
- [x] **This mini-blog on Hugging Face** — included for evaluator context and validation links.  
- [x] Environment on **HF Space** — URL above.  
- [x] README on GitHub links Space + materials.

---

## 6. Resources we actually used

From **OpenEnv_Hackathon_Resources.md**:

- OpenEnv: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [meta-pytorch.org/OpenEnv](https://meta-pytorch.org/OpenEnv/)  
- Hub collections: [huggingface.co/openenv](https://huggingface.co/openenv)  
- Tutorials: [OpenEnv/tutorial](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)  

Participant guide themes: RL loop, verifier-first rewards, GRPO/Unsloth stack, deploy Space early—we followed that order.

---

## 7. TL;DR (organizer quote)

> Build an environment an LLM can **train on** to get measurably better at something interesting. **Then show that training. Then tell the story.**

We built the MLOps simulator, wired **GRPO to live rewards**, and publish **metrics + plots**. Judges: start at the **Space URL**, then open the **model repo** for training artifacts if needed.

---

*Tags: `openenv`, `trl`, `unsloth`, `grpo`, `mlops`, `hackathon`*
