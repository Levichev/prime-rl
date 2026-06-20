"""Unroll a multi-turn prompt/completion SFT dataset into per-turn samples.

Each source row is a conversation: ``prompt`` (context ending at a user query)
plus ``completion`` (the rest of the trajectory: assistant + tool + follow-up
user turns). We split it into one sample per user query, where:

    prompt_out     = full history up to AND including that user message
    completion_out = the assistant response block for that user
                     (everything until the next user message)

Why: history-truncating chat templates (e.g. Nemotron-3
``truncate_history_thinking=True``) keep <think> reasoning only on the turn
answering the *last* user message. Feeding a whole multi-turn conversation as
one sample therefore supervises reasoning on the final turn only. Unrolling
makes every turn "current" in its own sample, and pairing this with
``[data.loss_mask] completion_only = true`` trains exactly that turn's response
(prompt = masked history) — matching inference token-for-token.

Single-user conversations (agentic single-query traces) yield exactly one
sample identical to the source, so this is safe to run on the whole dataset.

Usage:
    uv run python scripts/unroll_sft_dataset.py <in_dataset_dir> <out_dataset_dir>
"""

from __future__ import annotations

import argparse

from datasets import Dataset, DatasetDict, load_from_disk

MSG_KEYS = ("role", "content", "tool_calls")


def _norm_msg(m: dict) -> dict:
    """Keep only the canonical message keys with stable defaults so the output
    schema matches the source (and no transient masking tags leak in)."""
    return {
        "role": m.get("role"),
        "content": m.get("content") or "",
        "tool_calls": m.get("tool_calls") or [],
    }


def unroll_conversation(prompt: list[dict], completion: list[dict]) -> tuple[list[tuple[list[dict], list[dict]]], str | None]:
    """Return (samples, anomaly). ``samples`` is a list of (prompt_out, completion_out).

    anomaly is a short reason string when the whole conversation is skipped
    (currently only a stray non-leading ``system`` message, which the renderer
    rejects), else None.
    """
    full = [dict(m) for m in list(prompt) + list(completion)]

    # A system message is only valid at index 0. Anything else would crash the
    # renderer ("System message must be at the beginning."), so skip + report.
    sys_positions = [i for i, m in enumerate(full) if m.get("role") == "system"]
    if sys_positions and sys_positions != [0]:
        return [], "system_not_leading"

    user_idx = [i for i, m in enumerate(full) if m.get("role") == "user"]
    if not user_idx:
        return [], "no_user"

    ends = user_idx[1:] + [len(full)]
    samples: list[tuple[list[dict], list[dict]]] = []
    for u, end in zip(user_idx, ends):
        block = full[u + 1 : end]
        # Skip user turns with no assistant response (consecutive users, or a
        # trajectory that ends on a user message): nothing to supervise.
        if not any(m.get("role") == "assistant" for m in block):
            continue
        prompt_out = [_norm_msg(m) for m in full[: u + 1]]
        completion_out = [_norm_msg(m) for m in block]
        samples.append((prompt_out, completion_out))
    return samples, None


def unroll_split(ds: Dataset) -> tuple[Dataset, dict]:
    carry_cols = [c for c in ds.column_names if c not in ("prompt", "completion")]
    out_rows: list[dict] = []
    stats = {
        "in_rows": len(ds),
        "out_rows": 0,
        "skipped_convs": 0,
        "skipped_empty_blocks": 0,
        "src_assistant_turns": 0,
        "out_assistant_turns": 0,
        "anomalies": {},
        "skipped_indices": [],
    }
    for idx, ex in enumerate(ds):
        samples, anomaly = unroll_conversation(ex["prompt"], ex["completion"])
        if anomaly is not None:
            stats["skipped_convs"] += 1
            stats["anomalies"][anomaly] = stats["anomalies"].get(anomaly, 0) + 1
            stats["skipped_indices"].append(idx)
            continue
        full_roles = [m.get("role") for m in list(ex["prompt"]) + list(ex["completion"])]
        # Count source assistant turns only for non-skipped convs so the
        # conservation invariant (src == out) holds exactly.
        stats["src_assistant_turns"] += sum(r == "assistant" for r in full_roles)
        n_users = sum(r == "user" for r in full_roles)
        # users that produced no sample = empty response blocks (no assistant)
        stats["skipped_empty_blocks"] += max(0, n_users - len(samples))

        for prompt_out, completion_out in samples:
            stats["out_assistant_turns"] += sum(m["role"] == "assistant" for m in completion_out)
            row = {c: ex[c] for c in carry_cols}
            row["prompt"] = prompt_out
            row["completion"] = completion_out
            out_rows.append(row)

    stats["out_rows"] = len(out_rows)
    # Preserve the source schema/feature types and column order exactly.
    out = Dataset.from_list(out_rows, features=ds.features)
    out = out.select_columns(ds.column_names)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", help="input dataset dir (datasets.load_from_disk format)")
    ap.add_argument("out_dir", help="output dataset dir")
    args = ap.parse_args()

    obj = load_from_disk(args.in_dir)
    splits = obj if isinstance(obj, DatasetDict) else DatasetDict({"train": obj})

    out_dd = {}
    for name, ds in splits.items():
        out_ds, stats = unroll_split(ds)
        out_dd[name] = out_ds
        print(f"[{name}]")
        print(f"  in_rows                {stats['in_rows']}")
        print(f"  out_rows               {stats['out_rows']}")
        print(f"  skipped_convs          {stats['skipped_convs']} {stats['anomalies'] or ''}")
        if stats["skipped_indices"]:
            print(f"  skipped row indices    {stats['skipped_indices']}")
        print(f"  skipped_empty_blocks   {stats['skipped_empty_blocks']}")
        print(f"  assistant turns: src={stats['src_assistant_turns']} out={stats['out_assistant_turns']}")
        # Conservation: every assistant turn from non-skipped convs must survive.
        if stats["src_assistant_turns"] != stats["out_assistant_turns"]:
            raise SystemExit(
                f"  ASSERT FAILED: assistant turns not conserved "
                f"({stats['src_assistant_turns']} != {stats['out_assistant_turns']})"
            )
        print("  assistant-turn conservation: OK")

    DatasetDict(out_dd).save_to_disk(args.out_dir)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
