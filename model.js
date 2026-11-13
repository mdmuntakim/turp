/* Turp — tiny char-level neural language model (no dependencies).
 * Shared between Node (training) and the browser (inference).
 * Architecture: context chars -> embeddings -> ReLU hidden -> softmax over chars.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TurpModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var CONFIG = { context: 12, embed: 32, hidden: 192, seed: 1337 };

  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) {
    var u = Math.max(r(), 1e-9), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function createModel(chars) {
    var V = chars.length, C = CONFIG.context, E = CONFIG.embed, H = CONFIG.hidden;
    var r = rng(CONFIG.seed);
    var Ew = new Float32Array(V * E);
    for (var i = 0; i < Ew.length; i++) Ew[i] = randn(r) * 0.1;
    var Wh = new Float32Array(C * E * H);
    for (var j = 0; j < Wh.length; j++) Wh[j] = randn(r) * 0.1;
    var bh = new Float32Array(H);
    var Wo = new Float32Array(H * V);
    for (var k = 0; k < Wo.length; k++) Wo[k] = randn(r) * 0.02;
    var bo = new Float32Array(V);
    var params = V * E + C * E * H + H + H * V + V;
    return {
      config: { V: V, C: C, E: E, H: H },
      chars: chars.slice(),
      weights: { Ew: Ew, Wh: Wh, bh: bh, Wo: Wo, bo: bo },
      meta: { params: params, steps: 0, loss: Math.log(V), updated: new Date().toISOString() }
    };
  }

  function toJSON(m) {
    return JSON.stringify({
      config: m.config,
      chars: m.chars,
      weights: {
        Ew: Array.prototype.slice.call(m.weights.Ew),
        Wh: Array.prototype.slice.call(m.weights.Wh),
        bh: Array.prototype.slice.call(m.weights.bh),
        Wo: Array.prototype.slice.call(m.weights.Wo),
        bo: Array.prototype.slice.call(m.weights.bo)
      },
      meta: m.meta
    });
  }

  function fromJSON(src) {
    var o = typeof src === "string" ? JSON.parse(src) : src;
    return {
      config: o.config,
      chars: o.chars,
      weights: {
        Ew: Float32Array.from(o.weights.Ew),
        Wh: Float32Array.from(o.weights.Wh),
        bh: Float32Array.from(o.weights.bh),
        Wo: Float32Array.from(o.weights.Wo),
        bo: Float32Array.from(o.weights.bo)
      },
      meta: o.meta
    };
  }

  function indexMap(chars) {
    var m = Object.create(null);
    for (var i = 0; i < chars.length; i++) m[chars[i]] = i;
    return m;
  }

  /* One SGD step over a random batch. Returns average cross-entropy loss. */
  function trainStep(model, data, batchSize, lr) {
    var C = model.config.C, E = model.config.E, H = model.config.H, V = model.config.V;
    var Ew = model.weights.Ew, Wh = model.weights.Wh, bh = model.weights.bh;
    var Wo = model.weights.Wo, bo = model.weights.bo;
    var gEw = new Float32Array(Ew.length), gWh = new Float32Array(Wh.length);
    var gbh = new Float32Array(H), gWo = new Float32Array(Wo.length), gbo = new Float32Array(V);
    var e = new Float32Array(C * E), h = new Float32Array(H);
    var y = new Float32Array(V), p = new Float32Array(V), dh = new Float32Array(H);
    var n = data.length, totalLoss = 0, CE = C * E;

    for (var b = 0; b < batchSize; b++) {
      var start = (Math.random() * (n - C - 1)) | 0;
      for (var c = 0; c < C; c++) {
        var ch = data[start + c];
        for (var j = 0; j < E; j++) e[c * E + j] = Ew[ch * E + j];
      }
      for (var j2 = 0; j2 < H; j2++) {
        var s = bh[j2];
        for (var i2 = 0; i2 < CE; i2++) s += e[i2] * Wh[i2 * H + j2];
        h[j2] = s > 0 ? s : 0;
      }
      for (var k2 = 0; k2 < V; k2++) {
        var s2 = bo[k2];
        for (var j3 = 0; j3 < H; j3++) s2 += h[j3] * Wo[j3 * V + k2];
        y[k2] = s2;
      }
      var mx = -1e9;
      for (var k3 = 0; k3 < V; k3++) if (y[k3] > mx) mx = y[k3];
      var sum = 0;
      for (var k4 = 0; k4 < V; k4++) { p[k4] = Math.exp(y[k4] - mx); sum += p[k4]; }
      for (var k5 = 0; k5 < V; k5++) p[k5] /= sum;
      var target = data[start + C];
      totalLoss += -Math.log(Math.max(p[target], 1e-9));

      p[target] -= 1;
      for (var j4 = 0; j4 < H; j4++) dh[j4] = 0;
      for (var j5 = 0; j5 < H; j5++) {
        var hv = h[j5];
        for (var k6 = 0; k6 < V; k6++) {
          gWo[j5 * V + k6] += hv * p[k6];
          dh[j5] += Wo[j5 * V + k6] * p[k6];
        }
      }
      for (var k7 = 0; k7 < V; k7++) gbo[k7] += p[k7];
      for (var j6 = 0; j6 < H; j6++) {
        if (h[j6] <= 0) dh[j6] = 0;
        gbh[j6] += dh[j6];
        for (var i3 = 0; i3 < CE; i3++) gWh[i3 * H + j6] += e[i3] * dh[j6];
      }
      for (var i4 = 0; i4 < CE; i4++) {
        var de = 0;
        for (var j7 = 0; j7 < H; j7++) de += Wh[i4 * H + j7] * dh[j7];
        var pos = (i4 / E) | 0;
        var off = i4 - pos * E;
        gEw[data[start + pos] * E + off] += de;
      }
    }

    var scale = lr / batchSize;
    for (var a = 0; a < Ew.length; a++) Ew[a] -= scale * gEw[a];
    for (var a2 = 0; a2 < Wh.length; a2++) Wh[a2] -= scale * gWh[a2];
    for (var a3 = 0; a3 < H; a3++) bh[a3] -= scale * gbh[a3];
    for (var a4 = 0; a4 < Wo.length; a4++) Wo[a4] -= scale * gWo[a4];
    for (var a5 = 0; a5 < V; a5++) bo[a5] -= scale * gbo[a5];

    model.meta.steps += 1;
    model.meta.loss = totalLoss / batchSize;
    return model.meta.loss;
  }

  /* Sample a continuation of `prompt`. */
  function generate(model, prompt, opts) {
    opts = opts || {};
    var maxLen = opts.maxLen || 280;
    var temp = opts.temperature || 0.8;
    var topK = Math.min(opts.topK || 24, model.config.V);
    var C = model.config.C, E = model.config.E, H = model.config.H, V = model.config.V;
    var Ew = model.weights.Ew, Wh = model.weights.Wh, bh = model.weights.bh;
    var Wo = model.weights.Wo, bo = model.weights.bo;
    var map = indexMap(model.chars);
    var space = map[" "] !== undefined ? map[" "] : 0;

    var ctx = new Int32Array(C);
    var chars = Array.from(prompt);
    for (var i = 0; i < C; i++) {
      var ch = chars[chars.length - C + i];
      ctx[i] = ch && map[ch] !== undefined ? map[ch] : space;
    }

    var e = new Float32Array(C * E), h = new Float32Array(H), y = new Float32Array(V);
    var out = "";
    var CE = C * E;

    for (var step = 0; step < maxLen; step++) {
      for (var c = 0; c < C; c++) for (var j = 0; j < E; j++) e[c * E + j] = Ew[ctx[c] * E + j];
      for (var j2 = 0; j2 < H; j2++) {
        var s = bh[j2];
        for (var i2 = 0; i2 < CE; i2++) s += e[i2] * Wh[i2 * H + j2];
        h[j2] = s > 0 ? s : 0;
      }
      for (var k = 0; k < V; k++) {
        var s2 = bo[k];
        for (var j3 = 0; j3 < H; j3++) s2 += h[j3] * Wo[j3 * V + k];
        y[k] = s2 / temp;
      }
      var best = [];
      for (var k2 = 0; k2 < V; k2++) best.push(k2);
      best.sort(function (a, b) { return y[b] - y[a]; });
      best = best.slice(0, topK);
      var mx = -1e9, bi;
      for (var t = 0; t < best.length; t++) { bi = best[t]; if (y[bi] > mx) mx = y[bi]; }
      var sum = 0, probs = [];
      for (var t2 = 0; t2 < best.length; t2++) {
        var pr = Math.exp(y[best[t2]] - mx);
        probs.push(pr); sum += pr;
      }
      var r = Math.random() * sum, pick = best[0];
      for (var t3 = 0; t3 < best.length; t3++) { r -= probs[t3]; if (r <= 0) { pick = best[t3]; break; } }
      out += model.chars[pick];
      for (var m = 0; m < C - 1; m++) ctx[m] = ctx[m + 1];
      ctx[C - 1] = pick;
    }
    return out;
  }

  return {
    CONFIG: CONFIG,
    createModel: createModel,
    toJSON: toJSON,
    fromJSON: fromJSON,
    trainStep: trainStep,
    generate: generate,
    indexMap: indexMap
  };
});
