# Turp — a small AI trained from scratch on GitHub Actions

Turp is a tiny language model with **its own weights, its own trainer, and its own UI**.
Nothing is borrowed from existing models: every parameter is learned on a **GitHub Actions CPU**
from `corpus.txt`, checkpoint after checkpoint.

## How it works

| Part | File | Role |
|---|---|---|
| Model | `model.js` | Char-level neural net (context → embeddings → ReLU → softmax) shared by Node + browser |
| Trainer | `train.js` | Loads the last checkpoint, trains N steps, saves `model.json` |
| Corpus | `corpus.txt` | The only text Turp learns from |
| Training | `.github/workflows/train.yml` | Runs **every 3 hours** on GitHub Actions CPU and commits the new checkpoint |
| Deployment | `.github/workflows/deploy.yml` | Publishes the site to GitHub Pages on every push |
| UI | `index.html` | Loads `model.json` from the repo and chats with it in the browser |

## Live

**https://mdmuntakim.github.io/turp/**

## Reality check

Turp is ~93,000 parameters (under the 1-lakh cap by construction). That is a real neural
network, but it is a *small* one: it completes text and answers simple learned patterns, it
does not reason like ChatGPT. Making it bigger does not fit in a 100 MB file — file size ≈
parameters × bytes-per-parameter. At 4-bit, 100 MB ≈ 200M params max.

Everything runs on GitHub — training in Actions, inference in the visitor's browser. No phone,
no local server, no API key.

## Manual training run

```sh
node train.js --steps 1200 --batch 32 --lr 0.08
```

Or trigger **Actions → Train Turp → Run workflow**.
