#!/usr/bin/env python3
"""Turp trainer: a real GPT-2-architecture transformer trained from scratch on CPU.

The checkpoint is saved in Hugging Face format so llama.cpp's own converter can
turn it into a GGUF file (the same path every GGUF generative model takes).
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel

EOS = "<|endoftext|>"


def log(*a):
    print(*a, flush=True)


def build_tokenizer(text, out_dir, vocab_size):
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        (line for line in text.split("\n") if line.strip()),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
    )
    tok.save_model(str(out_dir))
    return tok


def load_tokenizer(model_dir):
    return ByteLevelBPETokenizer(
        str(Path(model_dir) / "vocab.json"),
        str(Path(model_dir) / "merges.txt"),
    )


def make_ids(tok, data_path, project_dir, repeat_qa=30, repeat_chat=500, repeat_code=300):
    story_text = Path(data_path).read_text(encoding="utf-8", errors="ignore")
    chunks = [c.strip() for c in story_text.split(EOS) if c.strip()]
    ids = []
    eos_id = tok.token_to_id(EOS)
    for i, chunk in enumerate(chunks):
        ids.extend(tok.encode(chunk).ids)
        ids.append(eos_id)
        if i % 20000 == 0:
            log(f"  tokenized {i}/{len(chunks)} stories")
    for name, times in (("qa.txt", repeat_qa), ("chat.txt", repeat_chat), ("code.txt", repeat_code)):
        p = Path(project_dir) / name
        if not p.exists():
            continue
        block = p.read_text(encoding="utf-8", errors="ignore").strip() + "\n\n"
        for _ in range(times):
            ids.extend(tok.encode(block).ids)
        ids.append(eos_id)
        log(f"  mixed {name} x{times}")
    return np.array(ids, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinystories.txt")
    ap.add_argument("--qa", default="qa.txt")
    ap.add_argument("--model", default="work/model")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--milestone", type=int, default=100000)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--ckpt-zip", default="work/turp-ckpt.zip")
    ap.add_argument("--status", default="status.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    model_dir = Path(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    state_path = model_dir / "turp_state.json"

    fresh = not (model_dir / "config.json").exists()
    if fresh:
        log("fresh start: training tokenizer")
        text = Path(args.data).read_text(encoding="utf-8", errors="ignore")
        tok = build_tokenizer(text, model_dir, args.vocab)
        eos_id = tok.token_to_id(EOS)
        vocab = tok.get_vocab_size()
        cfg = GPT2Config(
            vocab_size=vocab,
            n_positions=args.seq,
            n_ctx=args.seq,
            n_embd=136,
            n_layer=4,
            n_head=4,
            n_inner=544,
            layer_norm_epsilon=1e-5,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=eos_id,
            eos_token_id=eos_id,
            tie_word_embeddings=False,
        )
        model = GPT2LMHeadModel(cfg)
        state = {"steps": 0, "loss": None, "updated": None}
        log(f"new model: {sum(p.numel() for p in model.parameters()):,} params, vocab {vocab}")
    else:
        log("loading checkpoint")
        tok = load_tokenizer(model_dir)
        model = GPT2LMHeadModel.from_pretrained(str(model_dir))
        state = {"steps": 0, "loss": None, "updated": None}
        if state_path.exists():
            state = json.loads(state_path.read_text())
        log(f"resumed at {state.get('steps', 0)} steps")

    steps_done = int(state.get("steps", 0))
    in_segment = steps_done % args.milestone
    remaining = args.milestone - in_segment
    steps_to_run = max(1, min(args.steps, remaining))
    if steps_to_run != args.steps:
        log(f"milestone cap: {steps_to_run} steps to reach {steps_done + steps_to_run}")

    log("encoding corpus")
    ids = make_ids(tok, args.data, str(Path(__file__).resolve().parent))
    if len(ids) < args.seq + 2:
        raise SystemExit("corpus too small")
    log(f"corpus tokens: {len(ids):,}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    t0 = time.time()
    running = 0.0
    lo = 0
    hi = len(ids) - args.seq - 1
    best = state.get("loss")
    zip_path = Path(args.ckpt_zip)

    def persist(current_steps, loss_value):
        """Save model + status + zip atomically, so partial runs are never lost."""
        model.save_pretrained(str(model_dir))
        st = {
            "steps": int(current_steps),
            "loss": loss_value,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        state_path.write_text(json.dumps(st))
        try:
            cfg = json.loads((model_dir / "config.json").read_text())
            status = {
                "steps": st["steps"],
                "milestone": args.milestone,
                "loss": st["loss"],
                "updated": st["updated"],
                "params": int(sum(p.numel() for p in model.parameters())),
                "vocab": cfg["vocab_size"],
                "context": cfg["n_positions"],
                "arch": "gpt2",
            }
            Path(args.status).write_text(json.dumps(status, indent=1))
        except Exception as exc:  # status is cosmetic; checkpoint must still save
            log(f"status write failed: {exc}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_base = zip_path.parent / "ckpt-partial"
        tmp_zip = tmp_base.with_suffix(".zip")
        if tmp_zip.exists():
            tmp_zip.unlink()
        shutil.make_archive(str(tmp_base), "zip", root_dir=str(model_dir))
        os.replace(tmp_zip, zip_path)
        log(f"checkpoint saved at {st['steps']} steps ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return st

    for step in range(1, steps_to_run + 1):
        starts = np.random.randint(lo, hi, size=args.batch)
        batch = np.stack([ids[s:s + args.seq] for s in starts])
        x = torch.from_numpy(batch).long()
        out = model(input_ids=x, labels=x)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += float(loss)
        if step % 25 == 0 or step == 1:
            avg = running / (25 if step > 1 else 1)
            running = 0.0
            secs = time.time() - t0
            log(f"step {step}/{steps_to_run} loss {avg:.3f} {secs / step * 1000:.0f} ms/step")
            best = avg
        if step % 10000 == 0 and step != steps_to_run:
            persist(steps_done + step, best)

    steps_done += steps_to_run
    state = {
        "steps": steps_done,
        "loss": best,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model.eval()
    prompt = "User: hi bro\nTurp:"
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([tok.encode(prompt).ids]),
            max_new_tokens=48,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            pad_token_id=tok.token_to_id(EOS),
        )
    log("\n--- sample ---\n" + tok.decode(gen[0].tolist()) + "\n--------------")

    state = persist(steps_done, best)
    log(f"status: {json.dumps(state)}")


if __name__ == "__main__":
    main()
