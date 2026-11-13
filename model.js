/* Turp word-level neural language model (shared by Node trainer + browser inference).
 * Architecture: context of C tokens -> embeddings -> ReLU hidden layer -> softmax.
 * Trained with Adam. Weights can be loaded from / exported to GGUF.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TurpModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function paramCount(c) {
    return c.V * c.E + c.C * c.E * c.H + c.H + c.H * c.V + c.V;
  }

  function create(config) {
    var V = config.V, C = config.C, E = config.E, H = config.H;
    var CH = C * E;
    var Ew = new Float32Array(V * E);
    var Wh = new Float32Array(CH * H);
    var bh = new Float32Array(H);
    var Wo = new Float32Array(H * V);
    var bo = new Float32Array(V);
    var i;
    for (i = 0; i < Ew.length; i++) Ew[i] = (Math.random() * 2 - 1) * 0.1;
    var a = Math.sqrt(6 / (CH + H));
    for (i = 0; i < Wh.length; i++) Wh[i] = (Math.random() * 2 - 1) * a;
    var b = Math.sqrt(6 / (H + V));
    for (i = 0; i < Wo.length; i++) Wo[i] = (Math.random() * 2 - 1) * b;
    return { config: { V: V, C: C, E: E, H: H }, weights: { Ew: Ew, Wh: Wh, bh: bh, Wo: Wo, bo: bo }, adam: null };
  }

  function fromWeights(config, weights) {
    return {
      config: { V: config.V, C: config.C, E: config.E, H: config.H },
      weights: {
        Ew: weights.Ew instanceof Float32Array ? weights.Ew : Float32Array.from(weights.Ew),
        Wh: weights.Wh instanceof Float32Array ? weights.Wh : Float32Array.from(weights.Wh),
        bh: weights.bh instanceof Float32Array ? weights.bh : Float32Array.from(weights.bh),
        Wo: weights.Wo instanceof Float32Array ? weights.Wo : Float32Array.from(weights.Wo),
        bo: weights.bo instanceof Float32Array ? weights.bo : Float32Array.from(weights.bo)
      },
      adam: null
    };
  }

  function initAdam(m) {
    var W = m.weights;
    var st = { t: 0, m: {}, v: {} };
    for (var k in W) {
      st.m[k] = new Float32Array(W[k].length);
      st.v[k] = new Float32Array(W[k].length);
    }
    m.adam = st;
  }

  function forward(m, ids) {
    var c = m.config, C = c.C, E = c.E, H = c.H, V = c.V, CH = C * E;
    var W = m.weights;
    var x = new Float32Array(CH);
    for (var ci = 0; ci < C; ci++) {
      var id = ids[ci] < 0 ? 0 : ids[ci];
      var off = id * E;
      for (var e = 0; e < E; e++) x[ci * E + e] = W.Ew[off + e];
    }
    var h = new Float32Array(H);
    for (var j = 0; j < H; j++) {
      var s = W.bh[j];
      for (var k = 0; k < CH; k++) s += x[k] * W.Wh[k * H + j];
      h[j] = s > 0 ? s : 0;
    }
    var logits = new Float32Array(V);
    for (var v = 0; v < V; v++) {
      var t = W.bo[v];
      for (var jj = 0; jj < H; jj++) t += h[jj] * W.Wo[jj * V + v];
      logits[v] = t;
    }
    return { x: x, h: h, logits: logits };
  }

  function softmax(logits) {
    var V = logits.length, max = -Infinity, i;
    for (i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
    var probs = new Float32Array(V), sum = 0;
    for (i = 0; i < V; i++) { var e = Math.exp(logits[i] - max); probs[i] = e; sum += e; }
    for (i = 0; i < V; i++) probs[i] /= sum;
    return probs;
  }

  /* Train one Adam step on a batch (inputs: Int32Array B*C, targets: Int32Array B). */
  function trainStep(m, inputs, targets, B, lr) {
    var c = m.config, C = c.C, E = c.E, H = c.H, V = c.V, CH = C * E;
    var W = m.weights;
    if (!m.adam) initAdam(m);
    var st = m.adam;

    var g = m._grads;
    if (!g) {
      g = m._grads = {
        Ew: new Float32Array(W.Ew.length), Wh: new Float32Array(W.Wh.length),
        bh: new Float32Array(W.bh.length), Wo: new Float32Array(W.Wo.length),
        bo: new Float32Array(W.bo.length)
      };
    }
    for (var key in g) g[key].fill(0);

    var loss = 0;
    var x = new Float32Array(CH), h = new Float32Array(H), logits = new Float32Array(V);
    var dLogits = new Float32Array(V), dh = new Float32Array(H), dx = new Float32Array(CH);
    var scale = 1 / B;

    for (var b = 0; b < B; b++) {
      var base = b * C;
      for (var ci = 0; ci < C; ci++) {
        var id = inputs[base + ci];
        var eo = id * E;
        for (var e = 0; e < E; e++) x[ci * E + e] = W.Ew[eo + e];
      }
      for (var j = 0; j < H; j++) {
        var s = W.bh[j];
        for (var k = 0; k < CH; k++) s += x[k] * W.Wh[k * H + j];
        h[j] = s > 0 ? s : 0;
      }
      for (var v = 0; v < V; v++) {
        var t = W.bo[v];
        for (var j2 = 0; j2 < H; j2++) t += h[j2] * W.Wo[j2 * V + v];
        logits[v] = t;
      }
      var max = -Infinity, i;
      for (i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
      var sum = 0;
      for (i = 0; i < V; i++) { var ex = Math.exp(logits[i] - max); logits[i] = ex; sum += ex; }
      var target = targets[b];
      loss += -Math.log((logits[target] / sum) + 1e-9);

      for (i = 0; i < V; i++) dLogits[i] = logits[i] / sum;
      dLogits[target] -= 1;

      for (var j3 = 0; j3 < H; j3++) {
        var ds = 0, wo = j3 * V;
        for (var v2 = 0; v2 < V; v2++) ds += W.Wo[wo + v2] * dLogits[v2];
        dh[j3] = ds;
      }
      for (var j4 = 0; j4 < H; j4++) {
        g.bo[j4] += dLogits[j4] * scale;
        var pre = h[j4] > 0 ? dh[j4] * scale : 0;
        g.bh[j4] += pre;
        var wo2 = j4 * V;
        for (var v3 = 0; v3 < V; v3++) g.Wo[wo2 + v3] += h[j4] * dLogits[v3] * scale;
        for (var k2 = 0; k2 < CH; k2++) {
          g.Wh[k2 * H + j4] += x[k2] * pre;
          dx[k2] = pre * W.Wh[k2 * H + j4];
        }
        for (var k3 = 0; k3 < CH; k3++) {
          var tok = inputs[base + ((k3 / E) | 0)];
          g.Ew[tok * E + (k3 % E)] += dx[k3];
        }
      }
    }

    /* Adam update */
    st.t++;
    var b1 = 0.9, b2 = 0.999, eps = 1e-8;
    var bc1 = 1 - Math.pow(b1, st.t), bc2 = 1 - Math.pow(b2, st.t);
    for (var kName in W) {
      var w = W[kName], gr = g[kName], mm = st.m[kName], vv = st.v[kName];
      for (var p = 0; p < w.length; p++) {
        var gv = gr[p];
        mm[p] = b1 * mm[p] + (1 - b1) * gv;
        vv[p] = b2 * vv[p] + (1 - b2) * gv * gv;
        w[p] -= lr * (mm[p] / bc1) / (Math.sqrt(vv[p] / bc2) + eps);
      }
    }
    return loss / B;
  }

  function generate(m, promptIds, opts) {
    opts = opts || {};
    var maxNew = opts.maxNew || 60;
    var temperature = opts.temperature || 0.85;
    var topK = opts.topK || 40;
    var penalty = opts.repetitionPenalty || 1.15;
    var C = m.config.C, V = m.config.V;

    var ctx = [];
    for (var i = 0; i < promptIds.length; i++) ctx.push(promptIds[i]);
    if (ctx.length > C) ctx = ctx.slice(-C);

    var out = [];
    var recent = ctx.slice(-24);
    for (var step = 0; step < maxNew; step++) {
      while (ctx.length < C) ctx.unshift(0);
      var f = forward(m, ctx);
      var logits = f.logits;
      for (var r = 0; r < recent.length; r++) {
        var tid = recent[r];
        if (tid >= 0 && tid < V) logits[tid] = logits[tid] > 0 ? logits[tid] / penalty : logits[tid] * penalty;
      }
      for (var v = 0; v < V; v++) logits[v] /= temperature;
      var probs = softmax(logits);

      var order = [];
      for (var v2 = 0; v2 < V; v2++) order.push(v2);
      order.sort(function (a, b) { return probs[b] - probs[a]; });
      var cutoff = probs[order[Math.min(topK, V) - 1]];
      var total = 0;
      for (var v3 = 0; v3 < V; v3++) if (probs[v3] < cutoff) probs[v3] = 0; else total += probs[v3];
      var pick = Math.random() * total, acc = 0, chosen = 0;
      for (var v4 = 0; v4 < V; v4++) { acc += probs[v4]; if (acc >= pick) { chosen = v4; break; } }

      out.push(chosen);
      ctx.push(chosen);
      if (ctx.length > C) ctx.shift();
      recent.push(chosen);
      if (recent.length > 24) recent.shift();
    }
    return out;
  }

  return {
    paramCount: paramCount,
    create: create,
    fromWeights: fromWeights,
    forward: forward,
    softmax: softmax,
    trainStep: trainStep,
    generate: generate
  };
});
