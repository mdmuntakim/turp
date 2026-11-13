// GPT-2 forward pass over GGUF weights, with KV cache and sampling.
// This is the same transformer math llama.cpp runs; implemented here in plain JS.

import { getTensor } from "./gguf.js";
import { GPT2Tokenizer } from "./tokenizer.js";

function lnorm(x, w, b, out, eps) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  v /= n;
  const inv = 1 / Math.sqrt(v + eps);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

function matvec(W, rows, cols, x, out) {
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let s = 0;
    for (let c = 0; c < cols; c++) s += W[off + c] * x[c];
    out[r] = s;
  }
}

function addBias(out, b) {
  for (let i = 0; i < b.length; i++) out[i] += b[i];
}

function gelu(x) {
  const c = 0.7978845608028654;
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

export class GPT2 {
  constructor(gguf) {
    const md = gguf.metadata;
    this.md = md;
    this.nLayer = md["gpt2.block_count"];
    this.d = md["gpt2.embedding_length"];
    this.nHead = md["gpt2.attention.head_count"];
    this.ctx = md["gpt2.context_length"];
    this.ff = md["gpt2.feed_forward_length"];
    this.eps = md["gpt2.attention.layer_norm_epsilon"] ?? 1e-5;
    this.headDim = this.d / this.nHead;
    this.tokens = md["tokenizer.ggml.tokens"];
    this.vocabSize = this.tokens.length;
    this.tokenizer = new GPT2Tokenizer(this.tokens, md["tokenizer.ggml.merges"] || []);

    this.tokEmb = getTensor(gguf, "token_embd.weight").data;
    this.posEmb = getTensor(gguf, "position_embd.weight").data;
    this.outW = getTensor(gguf, "output.weight").data;
    this.outNormW = getTensor(gguf, "output_norm.weight").data;
    this.outNormB = getTensor(gguf, "output_norm.bias").data;

    this.layers = [];
    for (let i = 0; i < this.nLayer; i++) {
      const p = "blk." + i + ".";
      this.layers.push({
        ln1w: getTensor(gguf, p + "attn_norm.weight").data,
        ln1b: getTensor(gguf, p + "attn_norm.bias").data,
        qkvw: getTensor(gguf, p + "attn_qkv.weight").data,
        qkvb: getTensor(gguf, p + "attn_qkv.bias").data,
        ow: getTensor(gguf, p + "attn_output.weight").data,
        ob: getTensor(gguf, p + "attn_output.bias").data,
        ln2w: getTensor(gguf, p + "ffn_norm.weight").data,
        ln2b: getTensor(gguf, p + "ffn_norm.bias").data,
        w1: getTensor(gguf, p + "ffn_up.weight").data,
        b1: getTensor(gguf, p + "ffn_up.bias").data,
        w2: getTensor(gguf, p + "ffn_down.weight").data,
        b2: getTensor(gguf, p + "ffn_down.bias").data,
      });
    }

    this.kCache = this.layers.map(() => new Float32Array(this.ctx * this.d));
    this.vCache = this.layers.map(() => new Float32Array(this.ctx * this.d));

    this.x = new Float32Array(this.d);
    this.h = new Float32Array(this.d);
    this.qkv = new Float32Array(3 * this.d);
    this.att = new Float32Array(this.d);
    this.up = new Float32Array(this.ff);
    this.down = new Float32Array(this.d);
    this.logits = new Float32Array(this.vocabSize);
    this.scores = new Float32Array(this.ctx);
  }

  reset() {
    for (const k of this.kCache) k.fill(0);
    for (const v of this.vCache) v.fill(0);
  }

  forward(id, pos) {
    const d = this.d, ff = this.ff, H = this.nHead, hd = this.headDim;
    const x = this.x, h = this.h, qkv = this.qkv, att = this.att, up = this.up, down = this.down;

    const toff = id * d;
    for (let i = 0; i < d; i++) x[i] = this.tokEmb[toff + i] + this.posEmb[pos * d + i];

    for (let li = 0; li < this.nLayer; li++) {
      const L = this.layers[li];
      const kc = this.kCache[li], vc = this.vCache[li];

      lnorm(x, L.ln1w, L.ln1b, h, this.eps);
      matvec(L.qkvw, 3 * d, d, h, qkv);
      addBias(qkv, L.qkvb);

      const qo = 0, ko = d, vo = 2 * d;
      for (let i = 0; i < d; i++) {
        kc[pos * d + i] = qkv[ko + i];
        vc[pos * d + i] = qkv[vo + i];
      }

      const scale = 1 / Math.sqrt(hd);
      for (let head = 0; head < H; head++) {
        const base = head * hd;
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          const koff = t * d + ko + base;
          const qoff = qo + base;
          for (let j = 0; j < hd; j++) s += qkv[qoff + j] * kc[koff + j];
          s *= scale;
          this.scores[t] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) {
          const e = Math.exp(this.scores[t] - max);
          this.scores[t] = e;
          sum += e;
        }
        const inv = 1 / sum;
        for (let j = 0; j < hd; j++) att[base + j] = 0;
        for (let t = 0; t <= pos; t++) {
          const p = this.scores[t] * inv;
          const voff = t * d + vo + base;
          for (let j = 0; j < hd; j++) att[base + j] += p * vc[voff + j];
        }
      }

      matvec(L.ow, d, d, att, h);
      addBias(h, L.ob);
      for (let i = 0; i < d; i++) x[i] += h[i];

      lnorm(x, L.ln2w, L.ln2b, h, this.eps);
      matvec(L.w1, ff, d, h, up);
      addBias(up, L.b1);
      for (let i = 0; i < ff; i++) up[i] = gelu(up[i]);
      matvec(L.w2, d, ff, up, down);
      addBias(down, L.b2);
      for (let i = 0; i < d; i++) x[i] += down[i];
    }

    lnorm(x, this.outNormW, this.outNormB, h, this.eps);
    matvec(this.outW, this.vocabSize, d, h, this.logits);
    return this.logits;
  }

  sample(logits, temperature, topK, topP) {
    const n = logits.length;
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const arr = Array.from(idx);
    arr.sort((a, b) => logits[b] - logits[a]);
    const k = Math.min(topK, n);
    const max = logits[arr[0]];
    const probs = new Float32Array(k);
    let total = 0;
    for (let i = 0; i < k; i++) {
      const p = Math.exp((logits[arr[i]] - max) / Math.max(0.01, temperature));
      probs[i] = p;
      total += p;
    }
    let cutoff = k;
    if (topP < 1) {
      let cum = 0;
      for (let i = 0; i < k; i++) {
        cum += probs[i] / total;
        if (cum >= topP) { cutoff = i + 1; break; }
      }
    }
    let sum = 0;
    for (let i = 0; i < cutoff; i++) sum += probs[i];
    let r = Math.random() * sum;
    for (let i = 0; i < cutoff; i++) {
      r -= probs[i];
      if (r <= 0) return arr[i];
    }
    return arr[0];
  }

  async generate(prompt, opts = {}) {
    const maxTokens = opts.maxTokens ?? 64;
    const temperature = opts.temperature ?? 0.7;
    const topK = opts.topK ?? 40;
    const topP = opts.topP ?? 0.9;
    const stop = opts.stop ?? ["\nUser:"];
    const onToken = opts.onToken;
    const penalty = opts.repetitionPenalty ?? 1.25;

    this.reset();
    const ids = this.tokenizer.encode(prompt);
    if (!ids.length) throw new Error("empty prompt");
    let pos = 0;
    let logits = null;
    for (const id of ids) {
      if (pos >= this.ctx) break;
      logits = this.forward(id, pos);
      pos++;
    }
    const history = ids.slice();
    const out = [];
    for (let t = 0; t < maxTokens; t++) {
      if (pos >= this.ctx) break;
      const recent = new Set(history.slice(-64));
      for (const rid of recent) {
        const v = logits[rid];
        logits[rid] = v > 0 ? v / penalty : v * penalty;
      }
      const id = this.sample(logits, temperature, topK, topP);
      if (id === 0) break;
      out.push(id);
      history.push(id);
      const text = this.tokenizer.decode(out);
      if (onToken) onToken(text);
      if (stop.some((s) => text.endsWith(s))) break;
      logits = this.forward(id, pos);
      pos++;
      if (t % 4 === 3) await new Promise((r) => setTimeout(r, 0));
    }
    return this.tokenizer.decode(out);
  }
}

export function loadModel(gguf) {
  return new GPT2(gguf);
}
