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
import json
import os

from datasets import Dataset, DatasetDict, load_from_disk

# Per-worker tokenizer cache (datasets.map forks; load once per process).
_TOK = None
_TOK_NAME = None


def _get_tok(name: str):
    global _TOK, _TOK_NAME
    if _TOK is None or _TOK_NAME != name:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        _TOK_NAME = name
    return _TOK


def _tmpl_messages(msgs: list[dict]) -> list[dict]:
    """Shape messages for apply_chat_template (tool-call args -> dict)."""
    out = []
    for m in msgs:
        d = {"role": m["role"], "content": m.get("content") or ""}
        tcs = [c for c in (m.get("tool_calls") or []) if c]
        if tcs:
            fx = []
            for c in tcs:
                f = c.get("function") or c
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                fx.append({"function": {"name": f.get("name"), "arguments": a}})
            d["tool_calls"] = fx
        out.append(d)
    return out


def _prompt_render_len(tok, prompt_msgs: list[dict], tools: list[dict]) -> int:
    """Token length of the rendered prompt up to the first generated (trainable)
    token: history (reasoning stripped) + the assistant generation prompt. The
    segment's prompt always ends with a user, so standalone truncation matches
    the in-sample truncation exactly. L >= seq_len  <=>  0 trainable tokens."""
    text = tok.apply_chat_template(
        _tmpl_messages(prompt_msgs),
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        truncate_history_thinking=True,
    )
    return len(tok(text, add_special_tokens=False)["input_ids"])


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


def _strip_history_think(prompt_out: list[dict], completion_out: list[dict]) -> None:
    """In place: drop <think>…</think> from assistant turns before the last user
    (history), so a *stock* (non-stripping) chat template renders the same history
    a `truncate_history_thinking` template would. Current-turn reasoning (after the
    last user) is kept. Same rule as the modified Qwen template:
    ``content.split('</think>')[-1] | trim`` over the template's loop_messages
    (i.e. excluding a leading system message)."""
    full = prompt_out + completion_out
    offset = 1 if full and full[0].get("role") == "system" else 0
    loop_msgs = full[offset:]
    last_user = max((i for i, m in enumerate(loop_msgs) if m.get("role") == "user"), default=-1)
    for i, m in enumerate(loop_msgs):
        if m.get("role") == "assistant" and i < last_user:
            c = m.get("content") or ""
            if "</think>" in c:
                m["content"] = c.split("</think>")[-1].strip()


def coalesce_segments(
    prompt: list[dict], completion: list[dict], strip_history: bool = False
) -> list[tuple[list[dict], list[dict]]]:
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
        prompt_out = [_norm(m) for m in full[:s]]
        completion_out = [_norm(m) for m in block]
        if strip_history:
            _strip_history_think(prompt_out, completion_out)
        out.append((prompt_out, completion_out))
    return out


def _make_transform(carry_cols: list[str], tokenizer: str | None, seq_len: int | None, strip_history: bool):
    has_tools = "tools" in carry_cols

    def transform(batch: dict) -> dict:
        out: dict[str, list] = {"prompt": [], "completion": []}
        for c in carry_cols:
            out[c] = []
        tok = _get_tok(tokenizer) if tokenizer else None
        n = len(batch["prompt"])
        for r in range(n):
            segments = coalesce_segments(batch["prompt"][r], batch["completion"][r], strip_history=strip_history)
            tools = json.loads(batch["tools"][r] or "[]") if has_tools else []
            for prompt_out, completion_out in segments:
                # Skip segments whose trainable tokens fall entirely past seq_len
                # (0 trainable after chat-template render + truncation). Prompt
                # render length is monotonic across segments, so the first miss
                # means every later segment also misses -> stop this conversation.
                if tok is not None and _prompt_render_len(tok, prompt_out, tools) >= seq_len:
                    break
                out["prompt"].append(prompt_out)
                out["completion"].append(completion_out)
                for c in carry_cols:
                    out[c].append(batch[c][r])
        return out

    return transform


def split_dataset(ds: Dataset, workers: int, tokenizer: str | None, seq_len: int | None, strip_history: bool) -> Dataset:
    carry_cols = [c for c in ds.column_names if c not in ("prompt", "completion")]
    return ds.map(
        _make_transform(carry_cols, tokenizer, seq_len, strip_history),
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
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="if set (with --seq-len), drop segments with 0 trainable tokens after "
        "chat-template render + truncation to seq_len (e.g. Nemotron-3 tokenizer path)",
    )
    ap.add_argument("--seq-len", type=int, default=None, help="training context length (required with --tokenizer)")
    ap.add_argument(
        "--strip-history-think",
        action="store_true",
        help="remove <think>…</think> from history assistant turns in the data (turns before "
        "the last user), so a stock chat template + completion_only matches a "
        "truncate_history_thinking template at inference. Current-turn reasoning is kept.",
    )
    args = ap.parse_args()

    if (args.tokenizer is None) != (args.seq_len is None):
        ap.error("--tokenizer and --seq-len must be given together")

    obj = load_from_disk(args.in_dir)
    splits = obj if isinstance(obj, DatasetDict) else DatasetDict({"train": obj})

    out = {}
    for name, ds in splits.items():
        new = split_dataset(ds, args.workers, args.tokenizer, args.seq_len, args.strip_history_think)
        out[name] = new
        print(f"[{name}] {len(ds):,} conversations -> {len(new):,} examples")

    DatasetDict(out).save_to_disk(args.out_dir)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
