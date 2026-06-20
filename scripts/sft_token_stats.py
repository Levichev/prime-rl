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
      --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --workers 96

Parallelised across conversations with a process pool (the Jinja chat-template
render is GIL-bound, so processes — not threads — give the speedup). unroll@T
renders one growing-prefix sample per user turn, so a single long conversation
is ~quadratic in its turn count; pass --max-users to cap pathological tails.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os

from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

# Per-worker globals (set in the pool initializer so the tokenizer loads once
# per process, not once per task).
_TOK = None
_MAX_USERS = None


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
    """Per-user unroll: for each user turn, the message prefix up to and
    including that user's assistant response block. Mirrors unroll_sft_dataset.py
    (skips empty response blocks; [] for a stray non-leading system)."""
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


def ntok(messages, tools, trunc) -> int:
    text = _TOK.apply_chat_template(messages, tools=tools, tokenize=False, truncate_history_thinking=trunc)
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def _init_worker(tokenizer_name: str, max_users: int | None):
    global _TOK, _MAX_USERS
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _MAX_USERS = max_users


def process_row(task: tuple) -> tuple[str, dict]:
    """Compute one conversation's partial contribution to the per-subset stats."""
    sub, prompt, completion, tools_json = task
    msgs = to_template_messages(prompt, completion)
    tools = json.loads(tools_json or "[]")
    nusers = sum(m["role"] == "user" for m in msgs)

    p = dict(n=1, multi=int(nusers >= 2), src_F=0, src_T=0, unr_T=0, unr_rows=0,
             skipped=0, capped=0, asst=0, think_chars=0, content_chars=0, nothink_turns=0)

    for m in msgs:
        if m["role"] == "assistant":
            p["asst"] += 1
            c = m.get("content") or ""
            p["content_chars"] += len(c)
            if "<think>" in c and "</think>" in c:
                p["think_chars"] += len(c.split("</think>")[0].split("<think>")[-1])
            else:
                p["nothink_turns"] += 1

    p["src_F"] = ntok(msgs, tools, False)
    p["src_T"] = ntok(msgs, tools, True)

    if _MAX_USERS is not None and nusers > _MAX_USERS:
        p["capped"] = 1  # skip the expensive unroll for pathological tails
        return sub, p

    samples = unroll(msgs)
    if not samples and nusers >= 1:
        p["skipped"] = 1
        return sub, p
    for s in samples:
        p["unr_rows"] += 1
        p["unr_T"] += ntok(s, tools, True)
    return sub, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--split", default=None, help="split name if the dir is a DatasetDict")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="process pool size")
    ap.add_argument("--max-users", type=int, default=None,
                    help="skip unroll for conversations with more than this many user turns")
    args = ap.parse_args()

    obj = load_from_disk(args.dataset_dir)
    if isinstance(obj, DatasetDict):
        ds = obj[args.split] if args.split else obj[list(obj.keys())[0]]
    else:
        ds = obj

    # Materialise picklable tasks (RAM-cheap vs the render cost).
    tasks = [
        (ex.get("subset") or "None", ex["prompt"], ex["completion"], ex.get("tools"))
        for ex in ds
    ]
    print(f"loaded {len(tasks)} conversations; workers={args.workers}", flush=True)

    agg = collections.defaultdict(lambda: collections.Counter())
    # Sort longest-first so the heaviest conversations start early and the pool
    # isn't left waiting on one straggler at the end.
    tasks.sort(key=lambda t: -(len(t[1]) + len(t[2])))
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer, args.max_users)) as pool:
        for sub, p in tqdm(pool.imap_unordered(process_row, tasks, chunksize=1),
                           total=len(tasks), desc="rows", unit="conv"):
            agg[sub].update(p)

    print("\n#### Per-subset token volume ####")
    h = f"{'subset':<28}{'rows':>6}{'multi':>6}{'src@F':>13}{'src@T':>13}{'unroll@T':>13}{'unr/srcF':>10}"
    print(h); print("-" * len(h))
    T = collections.Counter()
    for sub, a in sorted(agg.items(), key=lambda kv: -kv[1]["unr_T"]):
        r = a["unr_T"] / a["src_F"] if a["src_F"] else 0
        for k in ("src_F", "src_T", "unr_T", "unr_rows", "n"):
            T[k] += a[k]
        print(f"{sub[:28]:<28}{a['n']:>6}{a['multi']:>6}{a['src_F']:>13,}{a['src_T']:>13,}{a['unr_T']:>13,}{r:>9.2f}x")
    print("-" * len(h))
    rr = T["unr_T"] / T["src_F"] if T["src_F"] else 0
    print(f"{'TOTAL':<28}{T['n']:>6}{'':>6}{T['src_F']:>13,}{T['src_T']:>13,}{T['unr_T']:>13,}{rr:>9.2f}x")

    print("\n#### Reasoning dropped by native truncate (src@F - src@T) ####")
    for sub, a in sorted(agg.items(), key=lambda kv: -(kv[1]["src_F"] - kv[1]["src_T"])):
        d = a["src_F"] - a["src_T"]; p = 100 * d / a["src_F"] if a["src_F"] else 0
        print(f"  {sub[:28]:<28} dropped={d:>13,}  ({p:>5.1f}% of subset)")

    print("\n#### Reasoning format ####")
    for sub, a in agg.items():
        tf = 100 * a["think_chars"] / a["content_chars"] if a["content_chars"] else 0
        print(f"  {sub[:28]:<28} asst_turns={a['asst']:>7}  no_think={a['nothink_turns']:>7}  "
              f"think={tf:>5.1f}% of content  unrolled_rows={a['unr_rows']}  "
              f"skipped={a['skipped']}  capped={a['capped']}")


if __name__ == "__main__":
    main()
