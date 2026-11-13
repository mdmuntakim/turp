/* Turp word-level tokenizer (shared by Node + browser) */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TurpTokenizer = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var TOKEN_RE = /\n|[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]/g;

  function tokenize(text) {
    var out = [];
    var m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(text)) !== null) out.push(m[0]);
    return out;
  }

  function buildVocab(text, size) {
    var counts = new Map();
    var toks = tokenize(text);
    for (var i = 0; i < toks.length; i++) {
      var t = toks[i];
      counts.set(t, (counts.get(t) || 0) + 1);
    }
    var words = Array.from(counts.keys());
    words.sort(function (a, b) {
      var d = counts.get(b) - counts.get(a);
      return d !== 0 ? d : a < b ? -1 : a > b ? 1 : 0;
    });
    if (size && words.length > size - 1) words = words.slice(0, size - 1);
    return ["<unk>"].concat(words);
  }

  function makeMap(vocab) {
    var m = new Map();
    for (var i = 0; i < vocab.length; i++) m.set(vocab[i], i);
    return m;
  }

  function encode(text, vocab) {
    var map = vocab instanceof Map ? vocab : makeMap(vocab);
    var toks = tokenize(text);
    var ids = new Int32Array(toks.length);
    for (var i = 0; i < toks.length; i++) {
      var id = map.get(toks[i]);
      ids[i] = id === undefined ? 0 : id;
    }
    return ids;
  }

  var NO_SPACE_BEFORE = { ".": 1, ",": 1, "!": 1, "?": 1, ":": 1, ";": 1, ")": 1, "]": 1, "}": 1, "%": 1, "’": 1, "'": 1, "\"": 1, "”": 1, "…": 1 };
  var OPENERS = { "(": 1, "[": 1, "{": 1, "“": 1, "\"": 1, "'": 1 };

  function decode(ids, vocab) {
    var out = "";
    var prev = "";
    for (var i = 0; i < ids.length; i++) {
      var t = vocab[ids[i]];
      if (t === undefined || t === "<unk>") continue;
      if (t === "\n") {
        out = out.replace(/[ \t]+$/, "") + "\n";
        prev = "\n";
        continue;
      }
      if (out === "" || prev === "\n" || prev === "" || OPENERS[prev]) out += t;
      else if (NO_SPACE_BEFORE[t]) out += t;
      else out += " " + t;
      prev = t;
    }
    return out.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  return { tokenize: tokenize, buildVocab: buildVocab, makeMap: makeMap, encode: encode, decode: decode };
});
