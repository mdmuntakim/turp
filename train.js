#!/usr/bin/env node
/* Turp trainer — runs on GitHub Actions CPU.
 * Loads an existing checkpoint (if any), continues training, saves model.json.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const Turp = require("./model.js");

const args = process.argv.slice(2);
function arg(name, def) {
  const i = args.indexOf("--" + name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : def;
}

const corpusPath = path.join(__dirname, "corpus.txt");
const modelPath = path.join(__dirname, "model.json");

const text = fs.readFileSync(corpusPath, "utf8");
const chars = Array.from(new Set(Array.from(text))).sort();

let model = null;
if (fs.existsSync(modelPath)) {
  try {
    const m = Turp.fromJSON(fs.readFileSync(modelPath, "utf8"));
    if (m.chars.join("") === chars.join("")) model = m;
    else console.log("vocabulary changed — starting fresh");
  } catch (e) {
    console.log("checkpoint unreadable — starting fresh:", e.message);
  }
}
if (!model) model = Turp.createModel(chars);

const data = new Uint32Array(Array.from(text).map(ch => model.chars.indexOf(ch)));

const steps = parseInt(arg("steps", "1500"), 10);
const batch = parseInt(arg("batch", "32"), 10);
const lr = parseFloat(arg("lr", "0.08"));

const t0 = Date.now();
let loss = model.meta.loss;
for (let i = 0; i < steps; i++) {
  loss = Turp.trainStep(model, data, batch, lr);
  if (i % 200 === 0) console.log(`step ${model.meta.steps}  loss ${loss.toFixed(4)}  ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

model.meta.loss = loss;
model.meta.updated = new Date().toISOString();
fs.writeFileSync(modelPath, Turp.toJSON(model));

const kb = (fs.statSync(modelPath).size / 1024).toFixed(0);
console.log(`saved ${modelPath}`);
console.log(`params=${model.meta.params}  steps=${model.meta.steps}  loss=${loss.toFixed(4)}  size=${kb} KB  vocab=${model.config.V}`);

// quick qualitative sample
console.log("\n--- sample ---");
console.log(Turp.generate(model, "User: What is Turp?\nTurp:", { maxLen: 160, temperature: 0.8, topK: 20 }));
