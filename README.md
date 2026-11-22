# What does a sentence cost in electrons?

Code and data for the NeuroTitan Labs post of the same name: https://neurotitan.in/blog/sentence-cost-in-electrons (published 2026-11-22).

The post follows one sentence, in 204 languages, from tokens to multiply-adds to transistors switching on SkyWater's open 130 nm process (sky130), and counts the electrons. Every number in the post comes from a file in this repo. `m6/popper_check.py` recomputes all of them; its last output is in `m6/popper_check_output.txt`.

## What is here

| Folder | Step | Main script | Main outputs |
|---|---|---|---|
| `m0-smoketest/` | Toolchain check: one inverter in ngspice | `inverter.cir` | (not a result) |
| `m1/` | Words to tokens: FLORES-200 devtest, 204 languages, 8 tokenizers, bootstrap intervals | `m1_tokens.py` | `summary.csv`, `run_info.json` |
| `m2/` | Tokens to arithmetic and bytes: Qwen3 multiply-add counts, checked with PyTorch's FLOP counter | `m2_ops.py` | `m2_ops.json` |
| `m3a/` | Transistors: sky130 inverter and 28-transistor full adder, 3 corners, many supplies; 8-bit multiply-add composed from 72 adders | `m3a_*.py` | `*_sweep.csv`, `mac.json`, `mep.json` |
| `m3b/` | Memory: a 128-cell SRAM column on SkyWater's SRAM devices; published off-chip figures | `m3b_sram_sweep.py` | `sram_sweep.csv`, `sources.md` |
| `m4/` | Server model and the test against Deng et al.'s measurements; the whole chain per language | `m4_deng_test.py`, `m4_chain.py` | `per_language.csv`, `deng_test.json`, `chain_per_language.csv` |
| `m5/` | Monte Carlo mismatch: 1,000 SRAM cells and 50 adders per corner and supply; red-team ledger | `m5_sram_mc.py`, `m5_logic_mc.py` | `*_mc_summary.csv`, `red_team.md` |
| `m7-validation/` | A first real GPU measurement: energy per language on a laptop RTX 4060, through Ollama, power read with nvidia-smi every 100 ms | `measure_ollama.py`, `analyze_run.py`, `make_m7_figures.py` | `run1/raw/`, `run1/analysis_ac/`, `run1/analysis_battery/`, `figures/` |
| `m6/` | Number check, figures, data for the interactive explorer | `popper_check.py`, `make_figures.py`, `make_explorer_data.py` | `figures/*.svg`, `explorer.json` |
| `configs/` | Qwen3 model configs (shapes only) | | |

Each folder's README says what was run, the checks applied, and what is known to be wrong or missing.

## Reproduce

Python 3.13 (we used 3.13.5 on Windows 11). Packages, at the versions we used:

```
pip install numpy==2.2.6 tiktoken==0.13.0 tokenizers==0.23.1 huggingface_hub torch==2.11.0
```

Only `m1` (tokenizers) and the check in `m2` (torch) need anything beyond the standard library.

The analysis steps run in seconds from the repo root:

```
python -I m2/m2_ops.py configs m1/summary.csv out/m2
python -I m4/m4_deng_test.py m1/summary.csv configs/Qwen3-8B.json m4/deng2026_table5.csv out/m4
python -I m4/m4_chain.py m1/summary.csv configs/Qwen3-8B.json out/m4
python -I m3a/m3a_mac_compose.py m3a/fulladder_sweep.csv out/m3a
python -I m6/popper_check.py
python -I m6/make_figures.py
```

We checked that these reproduce the committed `m2_ops.json`, `per_language.csv`, `deng_test.json`, `chain_per_language.csv` and `mac.json` exactly.

The heavy steps need external inputs that are not redistributed here:

- **Tokens (`m1`)**: FLORES-200 (`flores200_dataset.tar.gz`, SHA-256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`). Tokenizer files download from Hugging Face; the exact revisions are in `m1/run_info.json`.
- **Circuits (`m0`, `m3a`, `m3b`, `m5`)**: ngspice 47 (we used `ngspice_con.exe`) and the sky130A PDK, volare version `c6d73a35f524070e85faff4a6a9eef49553ebc2b`. Each script takes the ngspice binary and `sky130.lib.spice` as arguments. The Monte Carlo runs take hours: about 2.7 h for SRAM and 1.7 h for logic on a 16 GB laptop. `MC_WORKERS` limits parallel decks.
- **Prompt length (`m4/m4_belebele_prompt.py`)**: Belebele's English file (facebook/belebele, CC BY-SA 4.0) and Qwen3-8B's `tokenizer.json`.

## What these numbers are, and are not

- **Simulations, not measurements on a chip.** sky130 at 27 °C only, with no temperature sweep.
- **Lower bounds for arithmetic and on-chip memory.** AND gates, wires, registers, decoders and sense amplifiers are not counted.
- **The server model is a roofline with ideal bandwidth.** Its absolute times are about 2.5× too short, so only ratios between languages are used.
- **It ranks Deng et al.'s 117 languages at Spearman 0.94 with nothing fitted, but over-predicts 114 of 116 of them.** Mean context length alone ranks them equally well.
- **Known open items** are listed in `m5/red_team.md` and in the post.

If you find an error, open an issue or email neurotitancontact@gmail.com. We correct the post and note what changed.

## About the GPU measurement (`m7-validation/`)

The measured model is a private 4-bit (Q4_K_M) fine-tune of Qwen3.5-4B. Its Ollama tag is replaced by `qwen3.5-4b-finetune-q4km` in these files. The laptop's charger came out 26 minutes into run 1, so the plugged-in and battery requests are analysed separately (`REGIME=ac` or `battery`, `T_SPLIT=1795335053`), never together. Reproduce the analysis with:

```
T_SPLIT=1795335053 REGIME=ac python -I m7-validation/analyze_run.py m7-validation/run1/raw out/m7_ac
```

## Third-party data

- `m4/deng2026_table5.csv`: values from Table 5 of Deng et al., "The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference", arXiv 2606.21869v3 (EMNLP 2026), extracted from the arXiv HTML for reproducibility. Cite their paper if you use them.
- `configs/Qwen3-*.json`: model shapes from Qwen's published configs (Qwen3, Apache 2.0).
- Token counts in `m1/` are statistics computed on FLORES-200 (NLLB Team, Meta; CC BY-SA 4.0).
- Off-chip memory figures are quoted from the sources listed in `m3b/sources.md`. The source PDFs are not included.

## Licence

Code: MIT (`LICENSE`). Our data and figures: CC BY-SA 4.0 (`LICENSE-DATA.md`). Third-party material keeps its own terms, listed above.
