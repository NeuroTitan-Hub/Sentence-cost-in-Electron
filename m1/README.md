# M1: words to tokens, 2026-11-19

Status: done; spot-check passed. The raw per-sentence file (48 MB) is not included; `m1_tokens.py` regenerates it; `summary.csv` and `run_info.json` are here.

## Data and tokenizers
- FLORES-200 devtest, original release (`flores200_dataset.tar.gz`, the file the official FLORES-200 page links to), 204 languages x 1,012 sentences, CC BY-SA 4.0 per Meta's dataset card. SHA-256 b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6. FLORES+ (the maintained, corrected version) needs a Hugging Face login and agreement, so it is not used yet; switching later is a one-line change.
- Tokenizers: GPT-2, cl100k, o200k (tiktoken 0.13.0); Qwen3-8B, DeepSeek-V3, Mistral-Nemo, BLOOM (tokenizer.json, revisions in run_info.json); plus raw UTF-8 bytes. Content tokens only (no BOS/EOS or chat tokens).
- Not included: Llama 3.1 and Gemma 3 (licence gate on Hugging Face). Deng et al. use both; Qwen3-8B, their main model, is included.

## Method
Ratio to English = total tokens over the 1,012 sentences / English total. 95% interval: paired bootstrap over sentences (2,000 resamples, seed 20261119). Runtime about 60 s on the laptop.

## Results (ratio to English; 1.00 = same as English)
| Tokenizer | Median over 203 languages | Most expensive language | Odia [95% CI] | Pashto | Hindi |
|---|---|---|---|---|---|
| gpt2 | 2.56 | shn_Mymr 18.69 | 13.26 [13.10, 13.43] | 5.24 | 7.41 |
| cl100k | 2.26 | shn_Mymr 14.98 | 12.40 [12.24, 12.56] | 3.76 | 4.76 |
| o200k | 1.75 | sat_Olck 13.70 | 4.99 [4.93, 5.04] | 1.71 | 1.57 |
| qwen3 | 2.12 | shn_Mymr 10.43 | 9.70 [9.57, 9.83] | 2.80 | 4.42 |
| deepseek-v3 | 2.07 | taq_Tfng 10.15 | 5.55 [5.48, 5.61] | 2.52 | 2.80 |
| mistral-nemo | 1.93 | khm_Khmr 14.70 | 12.87 [12.69, 13.04] | 1.98 | 1.80 |
| bloom | 1.81 | sat_Olck 12.62 | 1.37 [1.35, 1.38] | 2.51 | 1.28 |
| utf8-bytes | 1.19 | shn_Mymr 3.96 | 2.74 [2.71, 2.76] | 1.63 | 2.55 |

Readings (for the team, not yet post copy):
- On Qwen3, the model Deng et al. measured most, Odia is the 2nd most expensive of 204 languages (9.7x English); only Shan is higher.
- Tokenizer choice matters as much as language: Odia is 1.37x on BLOOM but 12.9x on Mistral-Nemo.
- Pashto, Deng et al.'s most energy-expensive language, is only 1.7x (o200k) to 5.2x (GPT-2) in tokens here, 2.8x on Qwen3. So most of their 179x total-energy gap for Pashto cannot come from tokenization of the same text; it must come from longer generated answers and per-token cost. M2 and M4 have to account for that.

## Reproducibility check (passed)
- 8 of 8 spot counts (English, Odia, Pashto, Chinese x Qwen3, o200k) re-tokenized directly match the CSV.
- decode(encode(text)) == text for all 8.
- English is about 27 tokens per sentence on every subword tokenizer (26.6 to 27.5), a consistency check.
- Chinese below 1.0 on Qwen3, DeepSeek and BLOOM is expected (Chinese-heavy training data).
