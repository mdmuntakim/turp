# Turp — a real GGUF generative model, trained from scratch

Turp is a **GPT-2-architecture transformer** trained from scratch on a GitHub Actions CPU.
Its weights are exported to **`turp-v1.gguf`** by **llama.cpp's own converter**, and the page
runs that GGUF **directly in your browser** with a small JavaScript inference engine (`gguf.js` + `gpt2.js`).

There is no retrieval layer, no rule table, and no existing model — every token is generated
by the transformer.

**Live: https://mdmuntakim.github.io/turp/**
**Model: https://github.com/mdmuntakim/turp/releases/tag/turp-v1**

## Architecture

| | |
|---|---|
| Type | GPT-2 decoder-only transformer |
| Layers | 4 |
| Embedding | 136 |
| Heads | 4 |
| Feed-forward | 544 |
| Context | 128 tokens |
| Vocabulary | 4,096 BPE tokens (byte-level) |
| Parameters | ~2.03 million |
| Training data | TinyStories (public-domain-style synthetic stories) + chat patterns |
| Optimizer | AdamW, 100,000 steps per run |
| Export | f16 GGUF via `llama.cpp/convert_hf_to_gguf.py` |
| Browser runtime | JavaScript GGUF engine (`gguf.js`, `tokenizer.js`, `gpt2.js`) |

## How it works

1. `.github/workflows/train.yml` runs every 3 hours.
2. `train.py` loads the previous Hugging Face checkpoint from the `turp-v1` release, trains 100,000 steps, saves a new checkpoint.
3. llama.cpp converts the checkpoint to `turp-v1.gguf` (f16).
4. The GGUF and checkpoint are uploaded to the release; the GGUF is committed to `main`.
5. Pages deploys, and `index.html` loads the GGUF and generates locally with the JS engine.

Training stops exactly on each **100,000-step** milestone, uploads a snapshot, and the next
run continues from that checkpoint — not from scratch.

## Files

| File | Role |
|---|---|
| `train.py` | Tokenizer, GPT-2 training, checkpoint + status |
| `qa.txt` | Small chat-format corpus mixed into training |
| `index.html` | Monochrome UI, progress/ETA, generation |
| `gguf.js` / `tokenizer.js` / `gpt2.js` | GGUF reader, byte-level BPE, transformer inference |
| `.github/workflows/train.yml` | Scheduled training + GGUF conversion |

## Generative, not rule-based

There is no lookup table and no `if input == ...` anywhere in the runtime. The GGUF
file stores only learned weights. For every reply, the transformer reads the prompt,
computes a probability for each possible next token, samples one, appends it, and
repeats. `chat.txt`, `code.txt` and `qa.txt` are **training material** (like any LLM's
instruction data) — they shape the weights, they are not consulted at answer time.

A tiny model can overfit and repeat training lines; that is memorization, not a rule
engine, and it is why the corpora are mixed at moderate weight instead of repeated
thousands of times.

## Honest limits

This is a genuine generative transformer, but a tiny one trained for a short time. Early
checkpoints produce simple, imperfect English. Quality improves with every 100,000-step run.
It is not ChatGPT-level and will not reliably answer arbitrary questions.
