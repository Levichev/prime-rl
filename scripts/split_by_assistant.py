"""Split each prompt/completion conversation into one training example per
assistant turn.

For every assistant message in the reconstructed conversation
(prompt + completion):

    prompt_out     = every message before it (the full history)
    completion_out = that single assistant message

i.e. the completion is exactly the next assistant turn that follows the last
assistant already present in the prompt. The intervening user/tool messages
stay in the prompt as context. One conversation -> as many examples as it has
(context-preceded) assistant turns.

Pure structural transform (no tokenizer). Pair with the model's chat template
at train time; long examples are still truncated to seq_len by `cat` packing.

Usage:
    uv run python scripts/split_by_assistant.py <in_dataset_dir> <out_dataset_dir>
"""

from __future__ import annotations

import argparse

from datasets import Dataset, DatasetDict, load_from_disk


def _norm_msg(m: dict) -> dict:
    return {
        "role": m.get("role"),
        "content": m.get("content") or "",
        "tool_calls": m.get("tool_calls") or [],
    }


def split_conversation(prompt: list[dict], completion: list[dict]) -> tuple[list[tuple[list[dict], list[dict]]], str | None]:
    """Return (samples, status). One sample per assistant turn that has at least
    one preceding message. status is a skip reason for the whole conversation
    (stray non-leading system) or None."""
    full = [dict(m) for m in list(prompt) + list(completion)]

    sys_pos = [i for i, m in enumerate(full) if m.get("role") == "system"]
    if sys_pos and sys_pos != [0]:
        return [], "system_not_leading"

    samples: list[tuple[list[dict], list[dict]]] = []
    for i, m in enumerate(full):
        if m.get("role") != "assistant":
            continue
        if i == 0:
            # assistant with no preceding context -> nothing to condition on
            continue
        prompt_out = [_norm_msg(x) for x in full[:i]]
        completion_out = [_norm_msg(m)]
        samples.append((prompt_out, completion_out))
    return samples, None


def split_dataset(ds: Dataset) -> tuple[Dataset, dict]:
    carry_cols = [c for c in ds.column_names if c not in ("prompt", "completion")]
    out_rows: list[dict] = []
    stats = {
        "in_rows": len(ds),
        "out_rows": 0,
        "skipped_convs": 0,
        "skipped_leading_assistant": 0,
        "src_assistant_turns": 0,
        "anomalies": {},
        "skipped_indices": [],
    }
    for idx, ex in enumerate(ds):
        samples, status = split_conversation(ex["prompt"], ex["completion"])
        if status is not None:
            stats["skipped_convs"] += 1
            stats["anomalies"][status] = stats["anomalies"].get(status, 0) + 1
            stats["skipped_indices"].append(idx)
            continue
        full = list(ex["prompt"]) + list(ex["completion"])
        n_assistant = sum(m.get("role") == "assistant" for m in full)
        stats["src_assistant_turns"] += n_assistant
        # assistant turns with no preceding context are dropped (leading assistant)
        stats["skipped_leading_assistant"] += n_assistant - len(samples)
        for prompt_out, completion_out in samples:
            row = {c: ex[c] for c in carry_cols}
            row["prompt"] = prompt_out
            row["completion"] = completion_out
            out_rows.append(row)

    stats["out_rows"] = len(out_rows)
    out = Dataset.from_list(out_rows, features=ds.features)
    out = out.select_columns(ds.column_names)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", help="input dataset dir (datasets.load_from_disk format)")
    ap.add_argument("out_dir", help="output dataset dir")
    ap.add_argument("--show", type=int, default=1, help="print this many worked examples from the first conversation")
    args = ap.parse_args()

    obj = load_from_disk(args.in_dir)
    splits = obj if isinstance(obj, DatasetDict) else DatasetDict({"train": obj})

    out_dd = {}
    for name, ds in splits.items():
        # Worked example: show how the first non-skipped conversation splits.
        if args.show:
            for ex in ds:
                samples, status = split_conversation(ex["prompt"], ex["completion"])
                if status is None and samples:
                    print(f"[{name}] example conversation -> {len(samples)} samples:")
                    for k, (p, c) in enumerate(samples[: args.show]):
                        print(f"  sample {k}: prompt_roles={[m['role'] for m in p]}  "
                              f"completion_roles={[m['role'] for m in c]}")
                    break

        out_ds, stats = split_dataset(ds)
        out_dd[name] = out_ds
        print(f"[{name}]")
        print(f"  in_rows                     {stats['in_rows']}")
        print(f"  out_rows                    {stats['out_rows']}")
        print(f"  skipped_convs               {stats['skipped_convs']} {stats['anomalies'] or ''}")
        if stats["skipped_indices"]:
            print(f"  skipped row indices         {stats['skipped_indices']}")
        print(f"  dropped leading-assistant   {stats['skipped_leading_assistant']}")
        print(f"  src assistant turns         {stats['src_assistant_turns']}")
        # Each non-leading assistant turn becomes exactly one example.
        expected = stats["src_assistant_turns"] - stats["skipped_leading_assistant"]
        if stats["out_rows"] != expected:
            raise SystemExit(f"  ASSERT FAILED: out_rows {stats['out_rows']} != expected {expected}")
        print("  one-example-per-assistant: OK")

    DatasetDict(out_dd).save_to_disk(args.out_dir)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
