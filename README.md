# Turp — a word-level AI trained from scratch on GitHub Actions

Turp is its own language model: its own tokenizer, its own weights, its own GGUF file,
its own UI. Nothing is borrowed from an existing model. It learns on a **GitHub Actions CPU**
from public-domain English books, one checkpoint at a time.

**Live: https://mdmuntakim.github.io/turp/**
**Model release: https://github.com/mdmuntakim/turp/releases/tag/turp-v1**

## What kind of model is it?

| | |
|---|---|
| Type | Word-level neural language model (context → embeddings → ReLU → softmax) |
| Vocabulary | 4,000 tokens |
| Context | 16 tokens |
| Embedding / hidden | 64 / 320 |
| Parameters | ~1.9 million |
| File | `turp-v1.gguf` — real GGUF v3 container, ~7.5 MB |
| Trainer | Adam, 2,000 steps per run, batch 32 |
| Corpus | ~1.5 MB of public-domain English (Alice, Grimm, Andersen, Pride and Prejudice, Sherlock Holmes) + chat patterns |

The browser fetches **`turp-v1.gguf`** and parses it with `gguf.js` — the weights are not
stored as JSON.

## How the loop works

1. `.github/workflows/train.yml` runs every 3 hours (and on demand).
2. `train.js` loads the last `turp-v1.gguf`, trains 2,000 steps on the Actions CPU, writes a new checkpoint.
3. The checkpoint is committed to `main` (which triggers the Pages deploy) and uploaded to the `turp-v1` release.
4. `index.html` + `gguf.js` + `model.js` load the GGUF and generate in the visitor's browser.

## Files

| File | Role |
|---|---|
| `tokenizer.js` | Word/punctuation tokenizer and detokenizer |
| `model.js` | Forward pass, Adam training step, sampling |
| `gguf.js` | GGUF v3 writer/reader (Node + browser) |
| `train.js` | Incremental trainer |
| `corpus.txt` | The only training text |
| `index.html` | Black-and-white chat UI |

## Honest limits

A ~1.9M-parameter word-level model writes simple English and short continuations. It is
**not** ChatGPT-level and does not reason. More scheduled runs keep improving it.

On size: GitHub release assets allow up to **2 GB per file**, so uploading a 1.7 GB file is
possible. Training a 1.7 GB model (≈3.4B parameters at 4-bit) from scratch is not possible on
an Actions CPU — it would need roughly 70B training tokens and GPU-months. This model is the
largest size that can actually be trained to say real English on CPU in a scheduled run.
