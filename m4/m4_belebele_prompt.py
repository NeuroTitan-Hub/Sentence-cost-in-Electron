"""Measure the English Belebele prompt length in Qwen3 tokens, rebuilt from Deng et al.'s Appendix B template.

Template (arXiv 2606.21869v3, Appendix B, quoted in the README):
  <instruction, ending with the final line "#### N">
  Passage: <passage>
  Question: <question>
  Options: 1. <o1> 2. <o2> 3. <o3> 4. <o4>
  Let's think step by step.

Not stated in the paper, so measured both ways:
  - raw text vs Qwen3's chat template (user turn + assistant header);
  - the line layout (we join parts with newlines, and put each option on its own line).
Data: facebook/belebele, data/eng_Latn.jsonl, revision 7899cdfa, CC BY-SA 4.0, sha256 15af884c...
The chat wrapper is written out by hand in Qwen's ChatML form (user turn plus an open assistant turn), because
transformers is not installed and the cached snapshot has only tokenizer.json. It adds a fixed handful of tokens.
Usage: python -I m4_belebele_prompt.py <eng_Latn.jsonl> <Qwen3-8B tokenizer.json> <out json>
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

INSTRUCTION = ("Read the passage and answer the multiple-choice question. Reason step by step. "
               "Then, on a new last line, write the number of the correct option (1, 2, 3, or 4) "
               "in exactly this format:\n#### N")
PRIMER = "Let's think step by step."


def build(r: dict) -> str:
    opts = "\n".join(f"{i}. {r[f'mc_answer{i}']}" for i in range(1, 5))
    return (f"{INSTRUCTION}\n\nPassage: {r['flores_passage']}\n\nQuestion: {r['question']}\n\n"
            f"Options:\n{opts}\n\n{PRIMER}")


def summary(xs: list[int]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": st.median(xs),
            "p5": xs[int(0.05 * len(xs))], "p95": xs[int(0.95 * len(xs))], "min": xs[0], "max": xs[-1]}


def main() -> None:
    from tokenizers import Tokenizer
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    tk = Tokenizer.from_file(sys.argv[2])
    n = lambda text: len(tk.encode(text, add_special_tokens=False).ids)
    raw, chat, passage_only = [], [], []
    for r in rows:
        p = build(r)
        raw.append(n(p))
        chat.append(n(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"))
        passage_only.append(n(r["flores_passage"]))
    fixed = n(f"{INSTRUCTION}\n\n{PRIMER}")
    out = {"questions": len(rows), "distinct_passages": len({r["flores_passage"] for r in rows}),
           "raw_prompt_tokens": summary(raw), "chat_template_prompt_tokens": summary(chat),
           "passage_tokens": summary(passage_only), "instruction_plus_primer_tokens": fixed}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
