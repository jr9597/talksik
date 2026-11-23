import { readFileSync } from "node:fs";
import { logMel } from "./dsp.js";

const fixture = JSON.parse(readFileSync("/tmp/talksik-dsp.json"));
const audio = Float32Array.from(fixture.audio);
const { mel, nFrames, nMels } = logMel(audio, fixture);
let max = 0;
for (let i = 0; i < nFrames; i++) {
  for (let m = 0; m < nMels; m++) {
    const diff = Math.abs(mel[i * nMels + m] - fixture.mel[i][m]);
    if (diff > max) max = diff;
  }
}
console.log(`frames ${nFrames} max abs diff ${max}`);
if (max > 1e-3) {
  process.exit(1);
}
