# Turp — a real AI assistant that runs in your browser

Turp is a ChatGPT-style chat UI powered by a **real small language model** running
100% locally in the browser via **WebGPU** ([WebLLM](https://github.com/mlc-ai/web-llm)).

- No server, no API key, no rule-based responses — actual transformer inference.
- First load downloads the model once (cached afterwards).
- Works on desktop Chrome/Edge 113+ and WebGPU-enabled Chrome on Android.

## Models

| Model | Params | Footprint |
|---|---|---|
| SmolLM2 360M | 360M | ~580 MB VRAM |
| Qwen2.5 0.5B | 0.5B | ~945 MB VRAM |
| Qwen3 0.6B | 0.6B | ~1.9 GB VRAM |
| Llama 3.2 1B | 1.2B | ~1.1 GB VRAM |

## Why not 100,000 parameters?

A 100K-parameter model cannot behave like ChatGPT. Real assistants need hundreds of
millions to hundreds of billions of parameters — this is arithmetic, not a trick.
A tiny 0.5B model is already ~5,000× bigger than "1 lakh", and it still only
approximates ChatGPT-level fluency.

## Hosting

The GitHub Action deploys the static site to GitHub Pages on every push to `main`.

**Live:** https://mdmuntakim.github.io/turp/

## Note on the "action reset trick"

Public repositories get free standard GitHub Actions minutes — no reset trick is
needed or possible. Standard runners are CPU-only, which is why model inference
happens in the visitor's browser (WebGPU), not in Actions.
