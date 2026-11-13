/* Minimal GGUF v3 reader/writer for Turp (Node + browser).
 * Real GGUF container: metadata KV + tensor infos + aligned tensor data.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TurpGGUF = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var MAGIC = 0x46554747; // "GGUF"
  var VERSION = 3;
  var ALIGN = 32;
  var T_UINT32 = 4, T_FLOAT32 = 6, T_STRING = 8, T_ARRAY = 9;
  var GGML_F32 = 0;

  /* ---------- writer ---------- */
  function Writer() { this.chunks = []; this.length = 0; }
  Writer.prototype.push = function (bytes) { this.chunks.push(bytes); this.length += bytes.length; };
  Writer.prototype.u32 = function (v) {
    var b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v >>> 0, true); this.push(b);
  };
  Writer.prototype.u64 = function (v) {
    var b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); this.push(b);
  };
  Writer.prototype.f32 = function (v) {
    var b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.push(b);
  };
  Writer.prototype.str = function (s) {
    var b = new TextEncoder().encode(s);
    this.u64(b.length);
    this.push(b);
  };
  Writer.prototype.pad = function (align) {
    var p = (align - (this.length % align)) % align;
    if (p) this.push(new Uint8Array(p));
  };
  Writer.prototype.bytes = function () {
    var out = new Uint8Array(this.length), off = 0;
    for (var i = 0; i < this.chunks.length; i++) { out.set(this.chunks[i], off); off += this.chunks[i].length; }
    return out;
  };

  function build(spec) {
    var c = spec.config, vocab = spec.vocab, W = spec.weights, meta = spec.meta || {};
    var w = new Writer();
    var tensorNames = ["turp.embed.weight", "turp.hidden.weight", "turp.hidden.bias", "turp.output.weight", "turp.output.bias"];
    var keys = ["Ew", "Wh", "bh", "Wo", "bo"];
    var dims = [[c.V, c.E], [c.C * c.E, c.H], [c.H], [c.H, c.V], [c.V]];

    w.u32(MAGIC);
    w.u32(VERSION);
    w.u64(5);
    w.u64(11);

    w.str("general.architecture"); w.u32(T_STRING); w.str("turp");
    w.str("general.name"); w.u32(T_STRING); w.str("Turp");
    w.str("general.quantization_version"); w.u32(T_UINT32); w.u32(2);
    w.str("turp.context_length"); w.u32(T_UINT32); w.u32(c.C);
    w.str("turp.embedding_length"); w.u32(T_UINT32); w.u32(c.E);
    w.str("turp.hidden_length"); w.u32(T_UINT32); w.u32(c.H);
    w.str("turp.vocab_size"); w.u32(T_UINT32); w.u32(c.V);
    w.str("turp.training_steps"); w.u32(T_UINT32); w.u32(meta.steps || 0);
    w.str("turp.training_loss"); w.u32(T_FLOAT32); w.f32(meta.loss || 0);
    w.str("turp.updated"); w.u32(T_STRING); w.str(meta.updated || "");
    w.str("turp.vocab"); w.u32(T_ARRAY); w.u32(T_STRING); w.u64(vocab.length);
    for (var i = 0; i < vocab.length; i++) w.str(vocab[i]);

    /* tensor infos */
    var offsets = [], off = 0;
    for (var t = 0; t < keys.length; t++) {
      offsets.push(off);
      var bytes = W[keys[t]].length * 4;
      off += Math.ceil(bytes / ALIGN) * ALIGN;
    }
    for (var t2 = 0; t2 < keys.length; t2++) {
      w.str(tensorNames[t2]);
      w.u32(dims[t2].length);
      for (var d = 0; d < dims[t2].length; d++) w.u64(dims[t2][d]);
      w.u32(GGML_F32);
      w.u64(offsets[t2]);
    }
    w.pad(ALIGN);

    /* tensor data */
    for (var t3 = 0; t3 < keys.length; t3++) {
      var src = W[keys[t3]];
      var buf = new Uint8Array(src.length * 4);
      var dv = new DataView(buf.buffer);
      for (var j = 0; j < src.length; j++) dv.setFloat32(j * 4, src[j], true);
      w.push(buf);
      w.pad(ALIGN);
    }
    return w.bytes();
  }

  /* ---------- reader ---------- */
  function parse(input) {
    var u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
    var dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    var pos = 0;
    function u32() { var v = dv.getUint32(pos, true); pos += 4; return v; }
    function u64() { var v = Number(dv.getBigUint64(pos, true)); pos += 8; return v; }
    function f32() { var v = dv.getFloat32(pos, true); pos += 4; return v; }
    var dec = new TextDecoder();
    function str() { var n = u64(); var s = dec.decode(u8.subarray(pos, pos + n)); pos += n; return s; }
    function value(type) {
      switch (type) {
        case 0: pos += 1; return 0;
        case 1: pos += 1; return 0;
        case 2: { var a = dv.getUint16(pos, true); pos += 2; return a; }
        case 3: { var b = dv.getInt16(pos, true); pos += 2; return b; }
        case 4: return u32();
        case 5: { var c = dv.getInt32(pos, true); pos += 4; return c; }
        case 6: return f32();
        case 7: { var d = dv.getUint8(pos); pos += 1; return !!d; }
        case 8: return str();
        case 9: {
          var et = u32(), n = u64(), arr = [];
          for (var i = 0; i < n; i++) arr.push(value(et));
          return arr;
        }
        case 10: return u64();
        case 11: { var e = Number(dv.getBigInt64(pos, true)); pos += 8; return e; }
        case 12: { var f = dv.getFloat64(pos, true); pos += 8; return f; }
        default: throw new Error("unknown GGUF type " + type);
      }
    }

    var magic = u32();
    if (magic !== MAGIC) throw new Error("not a GGUF file");
    var version = u32();
    if (version !== VERSION) throw new Error("unsupported GGUF version " + version);
    var nTensors = u64(), nKv = u64();
    var kv = {};
    for (var i = 0; i < nKv; i++) {
      var k = str(), t = u32();
      kv[k] = value(t);
    }
    var infos = [];
    for (var j = 0; j < nTensors; j++) {
      var name = str(), nDims = u32(), dims = [];
      for (var d = 0; d < nDims; d++) dims.push(u64());
      var type = u32(), offset = u64();
      infos.push({ name: name, dims: dims, type: type, offset: offset });
    }
    var alignment = kv["general.alignment"] || ALIGN;
    var dataStart = Math.ceil(pos / alignment) * alignment;
    var nameToKey = {
      "turp.embed.weight": "Ew", "turp.hidden.weight": "Wh", "turp.hidden.bias": "bh",
      "turp.output.weight": "Wo", "turp.output.bias": "bo"
    };
    var weights = {};
    for (var t = 0; t < infos.length; t++) {
      var info = infos[t];
      var key = nameToKey[info.name];
      if (!key) continue;
      if (info.type !== GGML_F32) throw new Error("unsupported tensor type " + info.type);
      var numel = 1;
      for (var q = 0; q < info.dims.length; q++) numel *= info.dims[q];
      var start = dataStart + info.offset;
      var arr = new Float32Array(numel);
      for (var n = 0; n < numel; n++) arr[n] = dv.getFloat32(start + n * 4, true);
      weights[key] = arr;
    }
    var config = {
      V: kv["turp.vocab_size"], C: kv["turp.context_length"],
      E: kv["turp.embedding_length"], H: kv["turp.hidden_length"]
    };
    return {
      config: config,
      vocab: kv["turp.vocab"] || [],
      weights: weights,
      meta: {
        steps: kv["turp.training_steps"] || 0,
        loss: kv["turp.training_loss"] || 0,
        updated: kv["turp.updated"] || ""
      }
    };
  }

  return { build: build, parse: parse };
});
