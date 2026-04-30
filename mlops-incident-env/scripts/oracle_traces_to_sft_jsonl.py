"""
Convert oracle traces from inference.py (ORACLE_TRACE_PATH JSONL) into SFT examples for hf_train.py.

Each output line is {"messages": [...]} with system = hf_train SYS_PROMPT, user = last user turn
before the final oracle reply, assistant = hf_train-format diagnosis JSON (from final_diagnosis).

Usage:
  python scripts/oracle_traces_to_sft_jsonl.py traces/oracle.jsonl data/oracle_sft.jsonl
  # Then in Colab / HF Job:
  SFT_ORACLE_JSONL=data/oracle_sft.jsonl python hf_train.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def extract_hf_train_sys_prompt(hf_train_py: Path) -> str:
    text = hf_train_py.read_text(encoding="utf-8")
    m = re.search(r'SYS_PROMPT\s*=\s*"""(.*?)"""', text, flags=re.DOTALL)
    if not m:
        raise SystemExit(f"Could not find SYS_PROMPT = \"\"\"...\"\"\" in {hf_train_py}")
    return m.group(1).strip()


def last_user_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle traces → SFT JSONL for hf_train")
    ap.add_argument("input_jsonl", type=Path, help="Trace file from inference.py")
    ap.add_argument("output_jsonl", type=Path, help="SFT JSONL for SFT_ORACLE_JSONL")
    ap.add_argument(
        "--hf-train",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "hf_train.py",
        help="Path to hf_train.py (extract SYS_PROMPT)",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Extra filter on final_score (in addition to trace collection filter)",
    )
    args = ap.parse_args()

    sys_prompt = extract_hf_train_sys_prompt(args.hf_train)
    n_in, n_out = 0, 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.input_jsonl.open(encoding="utf-8") as inf, args.output_jsonl.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            if row.get("schema") != "oracle_trace_v1":
                continue
            score = float(row.get("final_score") or 0.0)
            if score < args.min_score:
                continue
            fd = row.get("final_diagnosis") or {}
            tgt = str(fd.get("target", "")).strip()
            if not tgt:
                continue
            user_txt = last_user_content(row.get("messages") or [])
            if not user_txt.strip():
                continue
            rc = str(fd.get("root_cause", ""))
            fx = str(fd.get("fix", ""))
            assistant = (
                "<think>Distilled from oracle agent trajectory.</think>\n"
                + json.dumps({"target": tgt, "root_cause": rc, "fix": fx}, ensure_ascii=False)
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_txt},
                {"role": "assistant", "content": assistant},
            ]
            outf.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read_lines={n_in} wrote_sft={n_out} -> {args.output_jsonl}")
    return 0 if n_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
