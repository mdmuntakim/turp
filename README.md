# Turp — a real GGUF generative model, trained from scratch

Turp is a **GPT-2-architecture transformer** trained from scratch on a GitHub Actions CPU.
Its weights are exported to **`turp-v1.gguf`** by **llama.cpp's own converter**, and the page
runs that GGUF **in your browser through wllama** (llama.cpp compiled to WebAssembly).

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
| Browser runtime | wllama (llama.cpp WASM), vendored in `wllama/` |

## How it works

1. `.github/workflows/train.yml` runs every 3 hours.
2. `train.py` loads the previous Hugging Face checkpoint from the `turp-v1` release, trains 100,000 steps, saves a new checkpoint.
3. llama.cpp converts the checkpoint to `turp-v1.gguf` (f16).
4. The GGUF and checkpoint are uploaded to the release; the GGUF is committed to `main`.
5. Pages deploys, and `index.html` loads the GGUF with wllama and generates locally.

Training stops exactly on each **100,000-step** milestone, uploads a snapshot, and the next
run continues from that checkpoint — not from scratch.

## Files

| File | Role |
|---|---|
| `train.py` | Tokenizer, GPT-2 training, checkpoint + status |
| `qa.txt` | Small chat-format corpus mixed into training |
| `index.html` | Monochrome UI, progress/ETA, wllama generation |
| `wllama/` | Vendored llama.cpp WASM runtime |
| `.github/workflows/train.yml` | Scheduled training + GGUF conversion |

## Honest limits

This is a genuine generative transformer, but a tiny one trained for a short time. Early
checkpoints produce simple, imperfect English. Quality improves with every 100,000-step run.
It is not ChatGPT-level and will not reliably answer arbitrary questions.
