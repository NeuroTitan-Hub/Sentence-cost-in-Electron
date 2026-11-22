"""M1: words to tokens.

Counts tokens for every FLORES-200 devtest language with several open tokenizers,
and the ratio to English for the same 1,012 sentences, with a paired bootstrap
95% interval over sentences.

Usage:
    python -I m1_tokens.py <flores200_dataset dir> <hf cache dir> <out dir>

Outputs:
    tokens_per_sentence.csv   one row per (tokenizer, language, sentence)
    summary.csv               one row per (tokenizer, language)
    run_info.json             versions, tokenizer sources, timing
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

SEED = 20261119
N_BOOT = 2000

# (name, kind, source). kind "tiktoken" uses a tiktoken encoding name,
# "hf" downloads only tokenizer.json from that Hugging Face repo, "bytes" counts UTF-8 bytes.
TOKENIZERS = [
    ("gpt2", "tiktoken", "gpt2"),
    ("cl100k", "tiktoken", "cl100k_base"),
    ("o200k", "tiktoken", "o200k_base"),
    ("qwen3", "hf", "Qwen/Qwen3-8B"),
    ("deepseek-v3", "hf", "deepseek-ai/DeepSeek-V3"),
    ("mistral-nemo", "hf", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("bloom", "hf", "bigscience/bloom"),
    ("utf8-bytes", "bytes", "UTF-8 byte count (ByT5-style byte model)"),
]


def load_tokenizer(kind: str, source: str, cache: Path):
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(source)
        return lambda texts: [len(t) for t in enc.encode_ordinary_batch(texts)], {"tiktoken": tiktoken.__version__}
    if kind == "hf":
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import tokenizers

        path = hf_hub_download(source, "tokenizer.json", cache_dir=str(cache))
        tok = Tokenizer.from_file(path)
        # Count content tokens only: no BOS/EOS or chat template tokens.
        return (
            lambda texts: [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)],
            {"tokenizers": tokenizers.__version__, "file": Path(path).name, "revision": Path(path).parent.name},
        )
    if kind == "bytes":
        return lambda texts: [len(t.encode("utf-8")) for t in texts], {}
    raise ValueError(kind)


def main() -> None:
    flores, cache, out = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((flores / "devtest").glob("*.devtest"))
    langs = {f.stem: f.read_text(encoding="utf-8").split("\n")[:1012] for f in files}
    assert all(len(v) == 1012 for v in langs.values()), "every language must have 1,012 sentences"
    chars = {k: np.array([len(s) for s in v]) for k, v in langs.items()}
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, 1012, size=(N_BOOT, 1012))

    info = {"python": platform.python_version(), "numpy": np.__version__, "seed": SEED, "n_boot": N_BOOT,
            "languages": len(langs), "tokenizers": {}}
    t0 = time.time()
    with open(out / "tokens_per_sentence.csv", "w", newline="", encoding="utf-8") as fs, \
         open(out / "summary.csv", "w", newline="", encoding="utf-8") as fsum:
        ws, wsum = csv.writer(fs), csv.writer(fsum)
        ws.writerow(["tokenizer", "lang", "sentence", "tokens", "chars"])
        wsum.writerow(["tokenizer", "lang", "total_tokens", "tokens_per_sentence", "chars_per_token",
                       "ratio_to_eng", "ratio_lo95", "ratio_hi95"])
        for name, kind, source in TOKENIZERS:
            count, meta = load_tokenizer(kind, source, cache)
            info["tokenizers"][name] = {"kind": kind, "source": source, **meta}
            counts = {k: np.array(count(v)) for k, v in langs.items()}
            eng = counts["eng_Latn"]
            eng_boot = eng[boot_idx].sum(axis=1)
            for lang, c in counts.items():
                for i, n in enumerate(c):
                    ws.writerow([name, lang, i, int(n), int(chars[lang][i])])
                ratio = c.sum() / eng.sum()
                rb = c[boot_idx].sum(axis=1) / eng_boot  # paired: same sentences resampled
                lo, hi = np.percentile(rb, [2.5, 97.5])
                wsum.writerow([name, lang, int(c.sum()), round(c.mean(), 3), round(chars[lang].sum() / c.sum(), 3),
                               round(ratio, 4), round(lo, 4), round(hi, 4)])
            print(f"{name:13s} done, {time.time() - t0:6.1f} s", flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
