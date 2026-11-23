/** Log-mel frontend. Must match train/audio.py. */

export function rfftPower(frame) {
  const n = frame.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = frame[i];
  fft(re, im);
  const bins = n / 2 + 1;
  const power = new Float64Array(bins);
  for (let i = 0; i < bins; i++) power[i] = re[i] * re[i] + im[i] * im[i];
  return power;
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextRe;
      }
    }
  }
}

export function loudest(audio, length) {
  const x = Float32Array.from(audio);
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= Math.max(1, x.length);
  for (let i = 0; i < x.length; i++) x[i] -= mean;
  if (x.length <= length) {
    const out = new Float32Array(length);
    const start = Math.floor((length - x.length) / 2);
    out.set(x, start);
    return out;
  }
  const hop = 160;
  const n = 1 + Math.floor((x.length - length) / hop);
  let best = 0;
  let bestE = -1;
  for (let i = 0; i < n; i++) {
    const start = i * hop;
    let e = 0;
    for (let j = 0; j < length; j++) e += x[start + j] * x[start + j];
    if (e > bestE) {
      bestE = e;
      best = start;
    }
  }
  return x.slice(best, best + length);
}

export function logMel(audio, meta) {
  const { window, hop, n_fft: nfft, filterbank } = meta;
  const win = meta.window_fn;
  const nFrames = 1 + Math.floor((audio.length - window) / hop);
  const nMels = filterbank.length;
  const mel = new Float32Array(nFrames * nMels);
  const frame = new Float64Array(nfft);
  for (let i = 0; i < nFrames; i++) {
    frame.fill(0);
    const start = i * hop;
    for (let j = 0; j < window; j++) frame[j] = audio[start + j] * win[j];
    const power = rfftPower(frame);
    for (let m = 0; m < nMels; m++) {
      let acc = 0;
      const row = filterbank[m];
      for (let k = 0; k < power.length; k++) acc += row[k] * power[k];
      const value = Math.log(Math.max(acc, 1e-10));
      mel[i * nMels + m] = (value - meta.mean[m]) / meta.std[m];
    }
  }
  return { mel, nFrames, nMels, raw: null };
}

export function spectrogram(audio, meta) {
  const { window, hop, n_fft: nfft } = meta;
  const win = meta.window_fn;
  const nFrames = 1 + Math.floor((audio.length - window) / hop);
  const bins = nfft / 2 + 1;
  const spec = new Float32Array(nFrames * bins);
  const frame = new Float64Array(nfft);
  let peak = -1e9;
  for (let i = 0; i < nFrames; i++) {
    frame.fill(0);
    const start = i * hop;
    for (let j = 0; j < window; j++) frame[j] = audio[start + j] * win[j];
    const power = rfftPower(frame);
    for (let k = 0; k < bins; k++) {
      const db = Math.log10(Math.max(power[k], 1e-10));
      spec[i * bins + k] = db;
      if (db > peak) peak = db;
    }
  }
  return { spec, nFrames, bins, peak };
}

export function resample(input, srcRate, dstRate) {
  if (srcRate === dstRate) return Float32Array.from(input);
  const length = Math.max(1, Math.round(input.length * dstRate / srcRate));
  const out = new Float32Array(length);
  const scale = srcRate / dstRate;
  for (let i = 0; i < length; i++) {
    const x = i * scale;
    const j = Math.floor(x);
    const t = x - j;
    const a = input[Math.min(j, input.length - 1)];
    const b = input[Math.min(j + 1, input.length - 1)];
    out[i] = a + (b - a) * t;
  }
  return out;
}
