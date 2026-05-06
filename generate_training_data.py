import os, json, time, re, random, argparse, requests
from pathlib import Path

ENV_URL     = os.getenv("HF_SPACE_URL", "https://jason9150-mlops-incident-env.hf.space")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
MODEL_NAME  = os.getenv("MODEL_NAME", "google/gemma-3-4b-it")
ORACLE_PATH = os.getenv("ORACLE_PATH", "oracle.jsonl")
ALL_TASKS   = ["easy", "medium", "hard", "cascade"]
def env_reset(task_id):
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action_type, target=None, diagnosis=None):
    payload = {"action_type": action_type}
    if target:    payload["target"] = target
    if diagnosis: payload.update(diagnosis)
    r = requests.post(f"{ENV_URL}/step", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def env_health():
    try:
        return requests.get(f"{ENV_URL}/health", timeout=10).status_code == 200
    except: return False
def call_gemma(prompt):
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN: headers["Authorization"] = f"Bearer {HF_TOKEN}"
    payload = {"inputs": prompt,
               "parameters": {"max_new_tokens": 300, "temperature": 0.3,
                               "do_sample": True, "return_full_text": False}}
    r = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
        headers=headers, json=payload, timeout=60)
    if r.status_code == 503:          # model loading
        time.sleep(20)
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()[0].get("generated_text", "").strip()
    return ""
def plan_steps(task_id, obs):
    comp = obs.get("component_status", {})
    def bad(s): return any(x in s.lower() for x in ["error","critical","warn","degrad"])

    if task_id == "easy":
        err = next((n for n,s in comp.items() if "error" in s.lower() and "pipeline" in n),
                   "data_pipeline_a")
        return [("inspect",err), ("query_logs",err),
                ("inspect","model_server"), ("check_metrics","feature_store"),
                ("query_logs","feature_store")]

    elif task_id == "medium":
        exclude = {"model_server","api_gateway"}
        primary = next((n for n,s in comp.items() if bad(s) and n not in exclude),
                       "feature_preprocessor_v2")
        return [("inspect",primary), ("compare_configs",primary),
                ("check_metrics","model_server"), ("inspect","load_balancer")]

    elif task_id == "hard":
        return [("check_feature_drift","feature_store"), ("check_metrics","model_server"),
                ("query_logs","model_server"), ("inspect","ab_testing_service"),
                ("query_logs","ab_testing_service")]

    elif task_id == "cascade":
        critical = [n for n,s in comp.items() if bad(s)][:3]
        steps = []
        for c in critical: steps += [("inspect",c), ("query_logs",c)]
        return steps[:8]
def build_prompt(task_id, obs, evidence):
    comp_text = "\n".join(f"[{v}] {k}" for k,v in obs.get("component_status",{}).items())
    ev_text   = "\n".join(evidence[-10:]) or "(no evidence yet)"
    return f"""You are SentinelAI, an expert MLOps incident response agent.

TASK: {task_id.upper()} -- {obs.get('goal', 'Diagnose the incident.')}

COMPONENT STATUS:
{comp_text}

EVIDENCE:
{ev_text}

OUTPUT a single JSON. Keys: action_type + target  (investigation)
OR action_type + target + root_cause + fix  (submit_diagnosis).

JSON:"""

def fallback_diagnosis(task_id, evidence):
    """Fallback diagnosis when Gemma is unavailable."""
    # Simple heuristic: find most common component in evidence
    components = {}
    for ev in evidence:
        for comp in ["data_pipeline_a", "feature_store", "model_server", "feature_preprocessor_v2", 
                     "load_balancer", "ab_testing_service", "embedding_service_v3"]:
            if comp in ev:
                components[comp] = components.get(comp, 0) + 1
    
    target = max(components, key=components.get) if components else "model_server"
    return {
        "action_type": "submit_diagnosis",
        "target": target,
        "root_cause": f"Detected issue in {target} based on investigation evidence",
        "fix": f"Review logs and metrics for {target}, consider rollback or restart"
    }

def run_episode(task_id, use_gemma=True):
    pairs, evidence = [], []
    obs   = env_reset(task_id)
    steps = plan_steps(task_id, obs)

    for i, (action_type, target) in enumerate(steps):
        prompt     = build_prompt(task_id, obs, evidence)
        completion = json.dumps({"action_type": action_type, "target": target})
        result     = env_step(action_type, target)
        feedback   = result.get("feedback", "")
        obs        = result.get("observation", obs)
        if feedback: evidence.append(f"> {action_type}({target})\n{feedback[:400]}")
        pairs.append({"prompt": prompt, "completion": completion,
                      "reward": result.get("reward", 0.0),
                      "task": task_id, "source": "fresh"})
        time.sleep(0.5)

    # final diagnosis
    final_prompt = build_prompt(task_id, obs, evidence) + "\n[FINAL: submit_diagnosis now]"
    action = call_gemma(final_prompt) if use_gemma and HF_TOKEN else None
    if not action: action = fallback_diagnosis(task_id, evidence)
    result = env_step("submit_diagnosis", target=action["target"], diagnosis=action)
    pairs.append({"prompt": final_prompt, "completion": json.dumps(action),
                  "reward": result.get("final_score", 0.0),
                  "task": task_id, "source": "fresh_diagnosis"})
    print(f"  final_score={result.get('final_score', 0):.4f}")
    return pairs
def parse_oracle(path):
    pairs = []
    if not Path(path).exists(): return pairs
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            msgs   = record.get("messages", [])
            task   = record.get("task_id", "unknown")
            for i in range(len(msgs)-1):
                u, a = msgs[i], msgs[i+1]
                if u.get("role")=="user" and a.get("role")=="assistant":
                    pairs.append({"prompt": u["content"], "completion": a["content"],
                                  "reward": a.get("reward", record.get("final_score", 0.0)),
                                  "task": task, "source": "oracle"})
    print(f"[oracle] {len(pairs)} pairs extracted")
    return pairs
FIELDS   = ["user_session_duration","transaction_amount","avg_order_value",
            "purchase_frequency_30d","click_through_rate","page_view_depth"]
TYPES    = [("FLOAT","STRING"),("INTEGER","FLOAT"),("STRING","INTEGER")]
VERSIONS = ["v2.1.0","v3.0.1","v2.3.1","v4.0.0"]

def augment(pairs, n=60):
    random.seed(42)
    out = []
    pool = [p for p in pairs if p["task"] in ("easy","medium")] or pairs
    for _ in range(n):
        base      = random.choice(pool)
        pr, co    = base["prompt"], base["completion"]
        f1, f2    = random.sample(FIELDS, 2)
        t1, t2    = random.choice(TYPES)
        v1, v2    = random.sample(VERSIONS, 2)
        for old, new in [(f1,f2),(t1,t2),(v1,v2)]:
            pr = pr.replace(old,new); co = co.replace(old,new)
        out.append({**base, "prompt":pr, "completion":co, "source":"augmented"})
    return out
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-per-task", type=int, default=12)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--no-gemma",   action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--oracle",  default=ORACLE_PATH)
    parser.add_argument("--output",  default="sentinel_grpo_dataset.jsonl")
    args = parser.parse_args()

    all_pairs = parse_oracle(args.oracle)   # Part 1

    if env_health():                         # Part 2
        fresh = []
        for task_id in args.tasks:
            for i in range(args.runs_per_task):
                print(f"  {task_id} run {i+1}/{args.runs_per_task}")
                fresh.extend(run_episode(task_id, use_gemma=not args.no_gemma))
                time.sleep(1.0)
        all_pairs.extend(fresh)

    if not args.no_augment:                  # Part 3
        all_pairs.extend(augment(all_pairs))

    # write output (deduplicated)
    seen, unique = set(), []
    for p in all_pairs:
        k = hash(p["prompt"][:200]+p["completion"][:200])
        if k not in seen: seen.add(k); unique.append(p)

    with open(args.output, "w") as f:
        for p in unique:
            f.write(json.dumps({"prompt":p["prompt"],"completion":p["completion"],
                                "reward":float(p.get("reward",0)),
                                "task":p["task"],"source":p["source"]}) + "\n")
    print(f"\nDone! {len(unique)} pairs -> {args.output}")

if __name__ == "__main__":
    main()
