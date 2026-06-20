"""Reasoning-preserving greedy split of a prompt/completion SFT dataset.

Splits each conversation into the *fewest* training examples such that every
reasoning turn is still trained with its `<think>` intact under a
history-truncating chat template (Nemotron-3 `truncate_history_thinking=True`)
+ `[data.loss_mask] completion_only = true`.

Rule: grow a completion segment by consuming turns; only cut before a `user`
message when the current segment already holds an *open* reasoning turn (a
non-empty `<think>` assistant turn after the segment's last user). No-reasoning
user blocks are absorbed into the same segment — stripping them is a no-op, so
no per-turn masking is needed.

Invariant: within every emitted completion, all reasoning turns sit after the
segment's last user -> the template keeps them; any stripped turn carries no
reasoning. `completion_only` masks the prompt. Each reasoning turn is therefore
trained exactly once, in its inference-time form.

Output schema == input schema (prompt / completion / carried columns).

Usage:
    uv run python scripts/coalesce_sft_dataset.py <in_dir> <out_dir> --workers 96
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, DatasetDict, load_from_disk


def _has_reasoning(content: str | None) -> bool:
    c = content or ""
    if "<think>" in c and "</think>" in c:
        return c.split("</think>")[0].split("<think>")[-1].strip() != ""
    return False


def _norm(m: dict) -> dict:
    return {
        "role": m.get("role"),
        "content": m.get("content") or "",
        "tool_calls": m.get("tool_calls") or [],
    }


def coalesce_segments(prompt: list[dict], completion: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    """Return [(prompt_out, completion_out), ...] for one conversation."""
    full = list(prompt) + list(completion)

    sys_pos = [i for i, m in enumerate(full) if m.get("role") == "system"]
    if sys_pos and sys_pos != [0]:
        return []  # stray non-leading system -> renderer would reject; skip conv

    first_a = next((i for i, m in enumerate(full) if m.get("role") == "assistant"), None)
    if first_a is None or first_a == 0:
        return []  # no assistant, or leading assistant with no history

    # Greedy segment boundaries (start, end) over `full`.
    segs: list[tuple[int, int]] = []
    start = first_a
    open_reasoning = False
    for i in range(first_a, len(full)):
        role = full[i].get("role")
        if role == "user":
            if open_reasoning:
                segs.append((start, i))   # close before this user (keeps its reasoning)
                start = i + 1             # this user goes into the next sample's prompt
            open_reasoning = False        # this user is now the segment's last user
        elif role == "assistant":
            if _has_reasoning(full[i].get("content")):
                open_reasoning = True
    segs.append((start, len(full)))

    out: list[tuple[list[dict], list[dict]]] = []
    for s, e in segs:
        block = full[s:e]
        if not any(m.get("role") == "assistant" for m in block):
            continue  # no trainable turn in this segment
        out.append(([_norm(m) for m in full[:s]], [_norm(m) for m in block]))
    return out


def _make_transform(carry_cols: list[str]):
    def transform(batch: dict) -> dict:
        out: dict[str, list] = {"prompt": [], "completion": []}
        for c in carry_cols:
            out[c] = []
        n = len(batch["prompt"])
        for r in range(n):
            for prompt_out, completion_out in coalesce_segments(batch["prompt"][r], batch["completion"][r]):
                out["prompt"].append(prompt_out)
                out["completion"].append(completion_out)
                for c in carry_cols:
                    out[c].append(batch[c][r])
        return out

    return transform


def split_dataset(ds: Dataset, workers: int) -> Dataset:
    carry_cols = [c for c in ds.column_names if c not in ("prompt", "completion")]
    return ds.map(
        _make_transform(carry_cols),
        batched=True,
        num_proc=workers,
        remove_columns=ds.column_names,
        features=ds.features,  # keep input schema/types exactly
        desc="coalesce-split",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", help="input dataset dir (datasets.load_from_disk format)")
    ap.add_argument("out_dir", help="output dataset dir")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="num_proc for datasets.map")
    args = ap.parse_args()

    obj = load_from_disk(args.in_dir)
    splits = obj if isinstance(obj, DatasetDict) else DatasetDict({"train": obj})

    out = {}
    for name, ds in splits.items():
        new = split_dataset(ds, args.workers)
        out[name] = new
        print(f"[{name}] {len(ds):,} conversations -> {len(new):,} examples")

    DatasetDict(out).save_to_disk(args.out_dir)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
