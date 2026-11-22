"""M7 experiment A (no-download version): real GPU energy per language.

The same task, on the same FLORES-200 devtest sentences, in many languages, on a local GPU through Ollama.
GPU board power is sampled with nvidia-smi (power.draw.instant) while each request runs. Standard library only.

Two tasks per language, both with the instruction in English and the text in the language:
  read  "Reply with only the word OK" after the text: nearly all work is prompt processing (prefill)
  copy  "Copy the text exactly": the answer is about as long as the text, in the same language (prefill + decode)

Per request we record:
  wall time, time to first token, Ollama's own prompt/eval counts and durations,
  GPU energy (trapezoid of power over the request), split at the first token into prefill and decode,
  energy above idle (minus idle watts times duration), peak VRAM, mean GPU utilisation, peak temperature.

Usage:  python -I measure_ollama.py <flores devtest dir> <out dir>
Env:    MODEL (default qwen3.5-4b-finetune-q4km), REPS (3), NSENT (10), LANGS (comma list), NUM_CTX (8192), MAXOUT (4096)
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL = os.environ["MODEL"]   # the Ollama model tag to measure (required)
REPS = int(os.environ.get("REPS", 3))
NSENT = int(os.environ.get("NSENT", 20))
READS = int(os.environ.get("READS", 5))   # read requests timed together as one window (one read is too short to sample)
NUM_CTX = int(os.environ.get("NUM_CTX", 12288))
MAXOUT = int(os.environ.get("MAXOUT", 6000))
LANGS = os.environ.get("LANGS", ",".join([
    "eng_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans", "jpn_Jpan", "kor_Hang",
    "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "ory_Orya", "amh_Ethi", "mya_Mymr", "khm_Khmr", "shn_Mymr",
])).split(",")
URL = "http://127.0.0.1:11434"
PROMPTS = {
    # every prompt starts with a unique run number, so Ollama cannot reuse a cached prefix from an earlier request
    "read": "Here is a text. Do not answer it. Reply with only the word OK.\n\n{text}",
    "copy": "Copy the following text exactly, in its original language, with no other words:\n\n{text}",
}
QUERY = "timestamp,power.draw.instant,utilization.gpu,memory.used,clocks.sm,temperature.gpu"


class Sampler:
    """Runs nvidia-smi in loop mode and keeps (epoch, W, util %, MiB, MHz, C) samples."""

    def __init__(self, ms: int = 100):
        self.samples: list[tuple[float, float, float, float, float, float]] = []
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits", f"-lms={ms}"],
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            try:
                ts, w, u, mem, clk, temp = [x.strip() for x in line.split(",")]
                t = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((t, float(w), float(u), float(mem), float(clk), float(temp)))
            except ValueError:
                continue

    def window(self, a: float, b: float):
        return [s for s in self.samples if a <= s[0] <= b]

    def energy(self, a: float, b: float) -> float:
        """Joules between a and b: trapezoid over samples, with the edges filled from the nearest samples."""
        s = self.samples
        if not s or b <= a:
            return 0.0
        pts = [(a, self._at(a))] + [(t, w) for t, w, *_ in s if a < t < b] + [(b, self._at(b))]
        return sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(pts, pts[1:]))

    def _at(self, t: float) -> float:
        best = min(self.samples, key=lambda x: abs(x[0] - t))
        return best[1]

    def stop(self):
        self.p.terminate()


NONCE = [0]


def chat(prompt: str, num_predict: int):
    NONCE[0] += 1
    prompt = f"[Request {NONCE[0]}] " + prompt
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True, "think": False,
                       "keep_alive": "30m",
                       "options": {"temperature": 0, "seed": 1, "num_ctx": NUM_CTX, "num_predict": num_predict}}).encode()
    req = urllib.request.Request(URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first, text, done = None, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            msg = json.loads(line)
            piece = msg.get("message", {}).get("content", "")
            if piece and t_first is None:
                t_first = time.time()
            text.append(piece)
            if msg.get("done"):
                done = msg
    t1 = time.time()
    return t0, t_first or t1, t1, "".join(text), done


def idle_watts(sampler: Sampler, seconds: float = 15) -> float:
    a = time.time()
    time.sleep(seconds)
    w = [x[1] for x in sampler.window(a, time.time())]
    return sum(w) / len(w)


def main():
    flores, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    texts = {}
    for L in LANGS:
        lines = (flores / f"{L}.devtest").read_text(encoding="utf-8").splitlines()[:NSENT]
        texts[L] = ("" if L.startswith(("zho", "jpn")) else " ").join(lines)
    sampler = Sampler()
    time.sleep(2)
    meta = {"model": MODEL, "reps": REPS, "nsent": NSENT, "num_ctx": NUM_CTX, "maxout": MAXOUT, "langs": LANGS,
            "prompts": PROMPTS, "started": datetime.now().isoformat(timespec="seconds"),
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,enforced.power.limit", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip(),
            "ollama": json.load(urllib.request.urlopen(URL + "/api/version"))["version"]}
    print("warm-up (loads the model)...", flush=True)
    chat(PROMPTS["read"].format(text=texts["eng_Latn"]), 1)
    time.sleep(3)
    meta["idle_w_start"] = idle_watts(sampler)
    print(f"idle {meta['idle_w_start']:.2f} W", flush=True)

    rows = []
    rng = random.Random(20261122)
    fields = ["rep", "lang", "task", "n_requests", "t0", "t_first", "t_end", "wall_s", "ttft_s", "prompt_tokens", "output_tokens",
              "prompt_eval_s", "eval_s", "load_s", "energy_j", "energy_prefill_j", "energy_decode_j", "idle_w",
              "energy_above_idle_j", "peak_vram_mib", "mean_util", "peak_temp_c", "n_samples", "out_chars", "in_chars",
              "copy_char_ratio", "done_reason"]
    f = open(out / "runs.csv", "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rep in range(1, REPS + 1):
        order = LANGS[:]
        rng.shuffle(order)
        for L in order:
            for task in ("read", "copy"):
                if task == "read":
                    # READS back-to-back requests in one window; per-request figures are the window divided by READS
                    res = [chat(PROMPTS[task].format(text=texts[L]), 1) for _ in range(READS)]
                    t0, tf, t1 = res[0][0], res[0][1], res[-1][2]
                    reply = res[-1][3]
                    ds = [r[4] for r in res]
                    d = {"prompt_eval_count": ds[0].get("prompt_eval_count"), "eval_count": ds[0].get("eval_count"),
                         "prompt_eval_duration": sum(x.get("prompt_eval_duration", 0) for x in ds),
                         "eval_duration": sum(x.get("eval_duration", 0) for x in ds),
                         "load_duration": sum(x.get("load_duration", 0) for x in ds), "done_reason": ds[-1].get("done_reason")}
                    n_req = READS
                else:
                    t0, tf, t1, reply, d = chat(PROMPTS[task].format(text=texts[L]), MAXOUT)
                    n_req = 1
                time.sleep(1.0)   # let the sampler catch up past t1
                win = sampler.window(t0, t1)
                idle = meta["idle_w_start"]
                e = sampler.energy(t0, t1)
                row = {"rep": rep, "lang": L, "task": task, "n_requests": n_req, "t0": round(t0, 3), "t_first": round(tf, 3), "t_end": round(t1, 3),
                       "wall_s": round(t1 - t0, 3), "ttft_s": round(tf - t0, 3),
                       "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
                       "prompt_eval_s": round(d.get("prompt_eval_duration", 0) / 1e9, 4), "eval_s": round(d.get("eval_duration", 0) / 1e9, 4),
                       "load_s": round(d.get("load_duration", 0) / 1e9, 4),
                       "energy_j": round(e, 3), "energy_prefill_j": round(sampler.energy(t0, tf), 3),
                       "energy_decode_j": round(sampler.energy(tf, t1), 3), "idle_w": round(idle, 3),
                       "energy_above_idle_j": round(e - idle * (t1 - t0), 3),
                       "peak_vram_mib": max((s[3] for s in win), default=None),
                       "mean_util": round(sum(s[2] for s in win) / len(win), 1) if win else None,
                       "peak_temp_c": max((s[5] for s in win), default=None), "n_samples": len(win),
                       "out_chars": len(reply), "in_chars": len(texts[L]),
                       "copy_char_ratio": round(len(reply) / len(texts[L]), 3) if task == "copy" else None,
                       "done_reason": d.get("done_reason")}
                w.writerow(row)
                f.flush()
                rows.append(row)
                print(f"rep {rep} {L:9s} {task:4s} in {row['prompt_tokens']:>5} out {row['output_tokens']:>5} "
                      f"wall {row['wall_s']:6.1f}s  E {row['energy_j']:7.1f} J (above idle {row['energy_above_idle_j']:7.1f})", flush=True)
                time.sleep(1.0)
    f.close()
    meta["idle_w_end"] = idle_watts(sampler)
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    sampler.stop()
    with open(out / "telemetry.csv", "w", newline="", encoding="utf-8") as g:
        tw = csv.writer(g)
        tw.writerow(["epoch", "power_w", "util_pct", "vram_mib", "sm_mhz", "temp_c"])
        tw.writerows(sampler.samples)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done", meta["finished"], f"idle end {meta['idle_w_end']:.2f} W", flush=True)


if __name__ == "__main__":
    main()
