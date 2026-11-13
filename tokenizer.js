// GPT-2 byte-level BPE tokenizer built from the vocab + merges stored in the GGUF.

const PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function byteToUnicode() {
  const bs = [];
  const cs = [];
  for (let i = 33; i <= 126; i++) { bs.push(i); cs.push(i); }
  for (let i = 161; i <= 172; i++) { bs.push(i); cs.push(i); }
  for (let i = 174; i <= 255; i++) { bs.push(i); cs.push(i); }
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2c = new Array(256);
  const c2b = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    b2c[bs[i]] = ch;
    c2b.set(ch, bs[i]);
  }
  return { b2c, c2b };
}

export class GPT2Tokenizer {
  constructor(tokens, merges) {
    this.tokens = tokens;
    this.vocab = new Map();
    tokens.forEach((t, i) => this.vocab.set(t, i));
    this.ranks = new Map();
    merges.forEach((m, i) => this.ranks.set(m, i));
    const { b2c, c2b } = byteToUnicode();
    this.b2c = b2c;
    this.c2b = c2b;
    this.eos = 0;
  }

  bpe(piece) {
    const bytes = encoder.encode(piece);
    let symbols = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) symbols[i] = this.b2c[bytes[i]];
    if (symbols.length < 2) return symbols;
    while (true) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.ranks.get(symbols[i] + " " + symbols[i + 1]);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIdx = i; }
      }
      if (bestIdx < 0) break;
      const merged = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols.splice(bestIdx, 2, merged);
    }
    return symbols;
  }

  encode(text) {
    const ids = [];
    for (const match of text.matchAll(PATTERN)) {
      const symbols = this.bpe(match[0]);
      for (const s of symbols) {
        const id = this.vocab.get(s);
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      const tok = this.tokens[id];
      if (tok === undefined) continue;
      if (tok === "<|endoftext|>") continue;
      for (const ch of tok) {
        const b = this.c2b.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return decoder.decode(new Uint8Array(bytes));
  }
}
