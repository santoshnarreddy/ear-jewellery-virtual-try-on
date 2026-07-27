#!/usr/bin/env node
/**
 * Node.js ONNX Runtime bench — measures load + forward latency and prints
 * the download size a browser user would need.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const MODEL = path.join(ROOT, "models/shgnet/hourglass_2stack.onnx");

function mb(n) {
  return (n / (1024 * 1024)).toFixed(2);
}

async function main() {
  if (!fs.existsSync(MODEL)) {
    console.error("Missing model:", MODEL);
    console.error("Run: python export_onnx.py");
    process.exit(1);
  }
  const modelBytes = fs.statSync(MODEL).size;

  console.log("=".repeat(64));
  console.log("Node ONNX Runtime — 2-SHGNet");
  console.log("=".repeat(64));
  console.log(`Model path : ${MODEL}`);
  console.log(`Model size : ${mb(modelBytes)} MB  ← user must download this`);
  console.log();
  console.log("Browser download budget (estimated):");
  console.log(`  ONNX weights     ${mb(modelBytes)} MB`);
  console.log(`  ort.min.js       ~0.3 MB`);
  console.log(`  ORT WASM runtime ~8–12 MB (first visit, then cached)`);
  console.log(
    `  TOTAL first visit ~${(modelBytes / (1024 * 1024) + 10).toFixed(0)}–${(
      modelBytes / (1024 * 1024) +
      15
    ).toFixed(0)} MB`
  );
  console.log();

  const tLoad0 = performance.now();
  const session = await ort.InferenceSession.create(MODEL, {
    executionProviders: ["cpu"],
  });
  console.log(`Session load: ${(performance.now() - tLoad0).toFixed(1)} ms`);
  console.log(`Providers   : ${session.handlers ? "cpu" : session}`);

  const inputName = session.inputNames[0];
  const x = new ort.Tensor(
    "float32",
    Float32Array.from({ length: 1 * 3 * 256 * 256 }, () => Math.random()),
    [1, 3, 256, 256]
  );

  // warmup
  for (let i = 0; i < 3; i++) {
    await session.run({ [inputName]: x });
  }
  const runs = 20;
  const t0 = performance.now();
  for (let i = 0; i < runs; i++) {
    await session.run({ [inputName]: x });
  }
  const ms = (performance.now() - t0) / runs;
  console.log(`Forward     : ${ms.toFixed(2)} ms  (${(1000 / ms).toFixed(1)} FPS)`);
  console.log("=".repeat(64));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
