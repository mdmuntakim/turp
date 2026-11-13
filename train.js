#!/usr/bin/env node
/* Incremental Turp trainer. Loads turp-v1.gguf (or starts fresh), trains, rewrites it. */
"use strict";
const fs = require("fs");
const path = require("path");
const { buildVocab, makeMap, encode, decode } = require("./tokenizer.js");
const TurpModel = require("./model.js");
const GGUF = require("./gguf.js");

const root = __dirname;
const CHECKPOINT = path.join(root, "turp-v1.gguf");
const CORPUS = path.join(root, "corpus.txt");
const MILESTONE = 100000;

function arg(name, def) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 ? Number(process.argv[i + 1]) : def;
}

const STEPS = arg("steps", 2000);
const BATCH = arg("batch", 32);
const LR = arg("lr", 0.0025);
const CONFIG = { V: 4000, C: 16, E: 64, H: 347 };

function main() {
  const text = fs.readFileSync(CORPUS, "utf8");
  let model, vocab, prevSteps = 0;

  if (fs.existsSync(CHECKPOINT)) {
    const ck = GGUF.parse(fs.readFileSync(CHECKPOINT));
    model = TurpModel.fromWeights(ck.config, ck.weights);
    vocab = ck.vocab;
    prevSteps = ck.meta.steps || 0;
    console.log(`loaded checkpoint: ${prevSteps} steps, ${vocab.length} vocab, ${TurpModel.paramCount(model.config).toLocaleString()} params`);
  } else {
    vocab = buildVocab(text, CONFIG.V);
    model = TurpModel.create({ V: vocab.length, C: CONFIG.C, E: CONFIG.E, H: CONFIG.H });
    console.log(`new model: ${TurpModel.paramCount(model.config).toLocaleString()} params, ${vocab.length} vocab`);
  }

  const map = makeMap(vocab);
  let ids = encode(text, map);
  console.log(`corpus: ${text.length.toLocaleString()} bytes, ${ids.length.toLocaleString()} tokens`);

  /* Stop exactly at each 100k milestone, then the next run continues from there. */
  const inSegment = prevSteps % MILESTONE;
  const remaining = MILESTONE - inSegment;
  const stepsToRun = Math.max(1, Math.min(STEPS, remaining));
  if (stepsToRun !== STEPS) console.log(`milestone cap: training ${stepsToRun} steps to reach ${prevSteps + stepsToRun}`);

  /* drop <unk>-heavy runs by keeping ids as-is; sampling avoids issues naturally */
  const inputs = new Int32Array(BATCH * model.config.C);
  const targets = new Int32Array(BATCH);
  const maxStart = ids.length - model.config.C - 1;
  let lossSum = 0, lossCount = 0, totalLoss = 0, totalSteps = 0;
  const t0 = Date.now();

  for (let step = 1; step <= stepsToRun; step++) {
    for (let b = 0; b < BATCH; b++) {
      const start = (Math.random() * maxStart) | 0;
      for (let c = 0; c < model.config.C; c++) inputs[b * model.config.C + c] = ids[start + c];
      targets[b] = ids[start + model.config.C];
    }
    const loss = TurpModel.trainStep(model, inputs, targets, BATCH, LR);
    lossSum += loss; lossCount++;
    totalLoss += loss; totalSteps++;
    if (step % 50 === 0 || step === 1) {
      const secs = (Date.now() - t0) / 1000;
      console.log(`step ${step}/${stepsToRun}  loss ${(lossSum / lossCount).toFixed(3)}  ${(secs / step * 1000).toFixed(0)} ms/step`);
      lossSum = 0; lossCount = 0;
    }
  }

  const prompt = "User: What is Turp?\nTurp:";
  const out = TurpModel.generate(model, Array.from(encode(prompt, map)), { maxNew: 60, temperature: 0.8, topK: 30 });
  console.log("\n--- sample ---\n" + decode(out, vocab) + "\n--------------");

  const meta = {
    steps: prevSteps + stepsToRun,
    milestone: MILESTONE,
    loss: totalLoss / Math.max(1, totalSteps),
    updated: new Date().toISOString()
  };
  const bytes = GGUF.build({ config: model.config, vocab: vocab, weights: model.weights, meta: meta });
  fs.writeFileSync(CHECKPOINT, Buffer.from(bytes));
  console.log(`wrote ${CHECKPOINT} (${(bytes.length / 1024 / 1024).toFixed(2)} MB), total steps ${meta.steps}`);
}

main();
