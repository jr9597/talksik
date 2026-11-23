import * as ort from "./vendor/ort.wasm.min.mjs";
import { logMel, loudest, resample, spectrogram } from "./dsp.js";

ort.env.wasm.wasmPaths = "./vendor/";

const hold = document.querySelector("#hold");
const hint = document.querySelector("#hint");
const canvas = document.querySelector("#spec");
const scores = document.querySelector("#scores");
const rows = document.querySelector("#rows");
const readout = document.querySelector("#readout");
const metrics = document.querySelector("#metrics");
const sampleBtn = document.querySelector("#sample");
const sampleName = document.querySelector("#sample-name");

const meta = await fetch("./model/meta.json").then((r) => r.json());
const session = await ort.InferenceSession.create("./model/talksik.onnx", {
  executionProviders: ["wasm"],
});

const hubert = meta.hubert_test_macro_f1 == null
  ? ""
  : ` Frozen HuBERT probe macro F1 ${meta.hubert_test_macro_f1.toFixed(2)}.`
metrics.textContent =
  `On-device encoder macro F1 ${meta.test_macro_f1.toFixed(2)}, balanced accuracy ${meta.test_balanced_accuracy.toFixed(2)}, ` +
  `on ${meta.test_clips} vowels from speakers it never trained on.${hubert}`

let exampleIndex = 0;
let recording = false;
let chunks = [];
let audioCtx;
let processor;
let source;
let stream;
let spaceDown = false;

drawSpec(new Float32Array(meta.target_samples), false);

sampleBtn.addEventListener("click", async () => {
  const example = meta.examples[exampleIndex % meta.examples.length];
  exampleIndex += 1;
  const flat = Float32Array.from(example.mel.flat());
  sampleName.textContent = `Clinic vowel labeled ${meta.display[example.label]}.`;
  await score(flat, meta.frames, meta.n_mels);
  readout.textContent =
    "Scored a stored clinic vowel, not your microphone. " + notADiagnosis(topLabel());
});

hold.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  hold.setPointerCapture(event.pointerId);
  start();
});
hold.addEventListener("pointerup", stop);
hold.addEventListener("pointercancel", stop);
window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) return;
  event.preventDefault();
  if (!spaceDown) {
    spaceDown = true;
    start();
  }
});
window.addEventListener("keyup", (event) => {
  if (event.code === "Space") {
    spaceDown = false;
    stop();
  }
});

async function start() {
  if (recording) return;
  recording = true;
  chunks = [];
  hold.classList.add("live");
  hold.querySelector(".label").textContent = "Listening";
  hint.textContent = "Keep a steady “ah” going.";
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  audioCtx = new AudioContext();
  source = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    if (recording) chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(audioCtx.destination);
}

async function stop() {
  if (!recording) return;
  recording = false;
  hold.classList.remove("live");
  hold.querySelector(".label").textContent = "Hold";
  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  const rate = audioCtx.sampleRate;
  await audioCtx.close();
  const merged = merge(chunks);
  if (merged.length < rate * 0.4) {
    hint.textContent = "That was too short. Hold a little longer.";
    return;
  }
  const audio = loudest(resample(merged, rate, meta.sample_rate), meta.target_samples);
  const { spec, nFrames, bins, peak } = spectrogram(audio, meta);
  paint(spec, nFrames, bins, peak);
  const features = logMel(audio, meta);
  await score(features.mel, features.nFrames, features.nMels);
  sampleName.textContent = "";
  hint.textContent = "Hold again to try another take.";
  readout.textContent = notADiagnosis(topLabel());
}

function merge(parts) {
  const length = parts.reduce((n, part) => n + part.length, 0);
  const out = new Float32Array(length);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

async function score(flat, frames, mels) {
  const input = new ort.Tensor("float32", flat, [1, frames, mels]);
  const out = await session.run({ mel: input });
  const logits = out.logits.data;
  const probs = softmax(logits);
  render(probs);
  scores.hidden = false;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function render(probs) {
  const order = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  rows.innerHTML = "";
  for (const [rank, item] of order.entries()) {
    const li = document.createElement("li");
    if (rank === 0) li.className = "top";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = meta.display[meta.labels[item.i]];
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${Math.round(item.p * 100)}%`;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${Math.max(2, item.p * 100)}%`;
    track.append(fill);
    li.append(name, pct, track);
    rows.append(li);
  }
}

function topLabel() {
  const top = rows.querySelector(".top .name");
  return top ? top.textContent : "this group";
}

function notADiagnosis(label) {
  return `Closest research group: ${label}. That is a similarity to the Saarbrücken clinic set, not a diagnosis.`;
}

function drawSpec(audio, live) {
  const { spec, nFrames, bins, peak } = spectrogram(audio, meta);
  paint(spec, nFrames, bins, peak, live);
}

function paint(spec, nFrames, bins, peak) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const image = ctx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    const bin = Math.min(bins - 1, Math.floor((1 - y / height) * bins));
    for (let x = 0; x < width; x++) {
      const frame = Math.min(nFrames - 1, Math.floor((x / width) * nFrames));
      const db = spec[frame * bins + bin];
      const t = Math.max(0, Math.min(1, (db - (peak - 4)) / 4));
      const o = (y * width + x) * 4;
      image.data[o] = 18 + t * 210;
      image.data[o + 1] = 16 + t * 70;
      image.data[o + 2] = 14 + t * 32;
      image.data[o + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}
