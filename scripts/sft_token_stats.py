"""Token statistics for a prompt/completion SFT dataset under the Nemotron-3
chat template: per-subset token volume for three scenarios, the reasoning
dropped by history truncation, and the cost of per-turn unrolling.

Scenarios (per subset):
  src@F     original, 1 sample/conversation, truncate_history_thinking=False
            (no cut — every turn keeps its <think>)
  src@T     original, 1 sample/conversation, truncate_history_thinking=True
            (Nemotron's native default — history <think> stripped)
  unroll@T  per-user unrolled, truncate_history_thinking=True
            (matches `scripts/unroll_sft_dataset.py` + completion_only)

Also reports: dropped reasoning (src@F - src@T), reasoning format breakdown,
and single- vs multi-user split.

Run:
  uv run python scripts/sft_token_stats.py <dataset_dir> \
      --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Heavy: unroll@T renders one growing-prefix sample per user turn, so cost is
~quadratic in turns for long conversations. Shows a tqdm progress bar.
"""

from __future__ import annotations

import argparse
import collections
import json

from datasets import DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def to_template_messages(prompt: list[dict], completion: list[dict]) -> list[dict]:
    """Concatenate and shape messages for apply_chat_template (tool-call
    arguments must be a dict, not the dataset's JSON string)."""
    out = []
    for m in list(prompt) + list(completion):
        d = {"role": m["role"], "content": m.get("content") or ""}
        tcs = [c for c in (m.get("tool_calls") or []) if c]
        if tcs:
            fixed = []
            for c in tcs:
                f = c.get("function") or c
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                fixed.append({"function": {"name": f.get("name"), "arguments": args}})
            d["tool_calls"] = fixed
        out.append(d)
    return out


def unroll(messages: list[dict]) -> list[list[dict]]:
    """Per-user unroll: for each user turn, return the message prefix up to and
    including that user's assistant response block. Mirrors unroll_sft_dataset.py
    (skips empty response blocks; returns [] for a stray non-leading system)."""
    sys_pos = [i for i, m in enumerate(messages) if m["role"] == "system"]
    if sys_pos and sys_pos != [0]:
        return []
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    ends = user_idx[1:] + [len(messages)]
    samples = []
    for u, end in zip(user_idx, ends):
        block = messages[u + 1 : end]
        if any(m["role"] == "assistant" for m in block):
            samples.append(messages[:end])
    return samples


def ntok(tok, messages, tools, trunc) -> int:
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, truncate_history_thinking=trunc)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--split", default=None, help="split name if the dir is a DatasetDict")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    obj = load_from_disk(args.dataset_dir)
    if isinstance(obj, DatasetDict):
        ds = obj[args.split] if args.split else obj[list(obj.keys())[0]]
    else:
        ds = obj

    agg = collections.defaultdict(lambda: dict(
        n=0, multi=0, src_F=0, src_T=0, unr_T=0, unr_rows=0, skipped=0,
        asst=0, think_chars=0, content_chars=0, nothink_turns=0,
    ))
    for ex in tqdm(ds, desc="rows", unit="conv"):
        sub = ex.get("subset") or "None"
        a = agg[sub]
        a["n"] += 1
        msgs = to_template_messages(ex["prompt"], ex["completion"])
        tools = json.loads(ex.get("tools") or "[]")
        nusers = sum(m["role"] == "user" for m in msgs)
        if nusers >= 2:
            a["multi"] += 1

        # reasoning format
        for m in msgs:
            if m["role"] == "assistant":
                a["asst"] += 1
                c = m.get("content") or ""
                a["content_chars"] += len(c)
                if "<think>" in c and "</think>" in c:
                    a["think_chars"] += len(c.split("</think>")[0].split("<think>")[-1])
                else:
                    a["nothink_turns"] += 1

        # token scenarios
        a["src_F"] += ntok(tok, msgs, tools, False)
        a["src_T"] += ntok(tok, msgs, tools, True)
        samples = unroll(msgs)
        if not samples and nusers >= 1:
            a["skipped"] += 1
            continue
        for s in samples:
            a["unr_rows"] += 1
            a["unr_T"] += ntok(tok, s, tools, True)

    print("\n#### Per-subset token volume ####")
    h = f"{'subset':<28}{'rows':>6}{'multi':>6}{'src@F':>12}{'src@T':>12}{'unroll@T':>12}{'unr/srcF':>10}"
    print(h); print("-" * len(h))
    T = collections.Counter()
    for sub, a in sorted(agg.items(), key=lambda kv: -kv[1]["unr_T"]):
        r = a["unr_T"] / a["src_F"] if a["src_F"] else 0
        for k in ("src_F", "src_T", "unr_T", "unr_rows", "n"):
            T[k] += a[k]
        print(f"{sub[:28]:<28}{a['n']:>6}{a['multi']:>6}{a['src_F']:>12,}{a['src_T']:>12,}{a['unr_T']:>12,}{r:>9.2f}x")
    print("-" * len(h))
    rr = T["unr_T"] / T["src_F"] if T["src_F"] else 0
    print(f"{'TOTAL':<28}{T['n']:>6}{'':>6}{T['src_F']:>12,}{T['src_T']:>12,}{T['unr_T']:>12,}{rr:>9.2f}x")

    print("\n#### Reasoning dropped by native truncate (src@F - src@T) ####")
    for sub, a in sorted(agg.items(), key=lambda kv: -(kv[1]["src_F"] - kv[1]["src_T"])):
        d = a["src_F"] - a["src_T"]; p = 100 * d / a["src_F"] if a["src_F"] else 0
        print(f"  {sub[:28]:<28} dropped={d:>12,}  ({p:>5.1f}% of subset)")

    print("\n#### Reasoning format ####")
    for sub, a in agg.items():
        tf = 100 * a["think_chars"] / a["content_chars"] if a["content_chars"] else 0
        print(f"  {sub[:28]:<28} asst_turns={a['asst']:>7}  no_think={a['nothink_turns']:>7}  "
              f"think={tf:>5.1f}% of content  unrolled_rows={a['unr_rows']}  skipped={a['skipped']}")


if __name__ == "__main__":
    main()
