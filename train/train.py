"""Train the encoder with a speaker-level split and export ONNX plus a model card."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio import N_MELS, hann, log_mel, mel_filterbank, n_frames  # noqa: E402
from model import VoiceEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "web" / "model"
LABELS = [
    "healthy",
    "hyperfunctional_dysphonia",
    "vocal_fold_paralysis",
    "laryngitis",
]
SEED = 7


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_speakers(manifest: list[dict]) -> dict[str, str]:
    rng = np.random.default_rng(SEED)
    by_label: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: set[tuple[str, str]] = set()
    for row in manifest:
        key = (row["label"], row["speaker"])
        if key in seen:
            continue
        seen.add(key)
        by_label[row["label"]].append(row["speaker"])
    assignment: dict[str, str] = {}
    for label, speakers in by_label.items():
        speakers = np.array(speakers)
        rng.shuffle(speakers)
        n = len(speakers)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        for speaker in speakers[:n_test]:
            assignment[f"{label}:{speaker}"] = "test"
        for speaker in speakers[n_test : n_test + n_val]:
            assignment[f"{label}:{speaker}"] = "val"
        for speaker in speakers[n_test + n_val :]:
            assignment[f"{label}:{speaker}"] = "train"
    return assignment


def augment(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = 10 ** (rng.uniform(-6, 6) / 20)
    wave = wave * gain
    shift = int(rng.integers(-1600, 1601))
    wave = np.roll(wave, shift)
    if shift > 0:
        wave[:shift] = 0
    elif shift < 0:
        wave[shift:] = 0
    return wave.astype(np.float32)


def batch_mel(
    waves: np.ndarray,
    filters: np.ndarray,
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    mels = np.stack([log_mel(wave, filters, window) for wave in waves])
    return ((mels - mean) / std).astype(np.float32)


def confusion(pred: np.ndarray, gold: np.ndarray, n: int) -> list[list[int]]:
    matrix = np.zeros((n, n), dtype=int)
    for g, p in zip(gold, pred):
        matrix[int(g), int(p)] += 1
    return matrix.tolist()


def macro_f1(matrix: np.ndarray) -> float:
    scores = []
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores))


def evaluate(
    model: nn.Module,
    waves: np.ndarray,
    labels: np.ndarray,
    filters: np.ndarray,
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    dev: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(waves), 64):
            mel = batch_mel(waves[start : start + 64], filters, window, mean, std)
            logits = model(torch.from_numpy(mel).to(dev))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    matrix = np.array(confusion(pred, labels, len(LABELS)))
    return macro_f1(matrix), pred, matrix


def main() -> None:
    torch.manual_seed(SEED)
    manifest = json.loads((CACHE / "manifest.json").read_text())
    audio = np.load(CACHE / "audio.npy")
    assignment = split_speakers(manifest)
    label_index = {label: i for i, label in enumerate(LABELS)}

    groups = {"train": [], "val": [], "test": []}
    for i, row in enumerate(manifest):
        split = assignment[f"{row['label']}:{row['speaker']}"]
        groups[split].append(i)

    filters = mel_filterbank()
    window = hann()
    train_idx = np.array(groups["train"])
    # Normalization from the training speakers only.
    sample = train_idx[: min(512, len(train_idx))]
    sample_mels = np.stack([log_mel(audio[i], filters, window) for i in sample])
    mean = sample_mels.mean(axis=(0, 1)).astype(np.float32)
    std = sample_mels.std(axis=(0, 1)).astype(np.float32)
    std = np.maximum(std, 1e-4)

    dev = device()
    print(f"device {dev} clips train/val/test "
          f"{len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])}", flush=True)
    model = VoiceEncoder().to(dev)
    counts = np.bincount([label_index[manifest[i]["label"]] for i in train_idx], minlength=4)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=dev)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    teacher = np.load(CACHE / "teacher_logits.npy") if (CACHE / "teacher_logits.npy").exists() else None

    best_f1 = -1.0
    best_state = None
    patience = 0
    for epoch in range(1, 41):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        seen = 0
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            waves = np.stack([augment(audio[i], rng) for i in batch])
            mel = batch_mel(waves, filters, window, mean, std)
            gold = torch.tensor(
                [label_index[manifest[i]["label"]] for i in batch],
                dtype=torch.long,
                device=dev,
            )
            opt.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(mel).to(dev))
            loss = loss_fn(logits, gold)
            if teacher is not None:
                target = torch.from_numpy(teacher[batch]).to(dev)
                log_p = logits.log_softmax(dim=-1)
                distill = -(target.softmax(dim=-1) * log_p).sum(dim=-1).mean()
                loss = loss + distill
            loss.backward()
            opt.step()
            total += float(loss) * len(batch)
            seen += len(batch)
        val_waves = audio[groups["val"]]
        val_labels = np.array([label_index[manifest[i]["label"]] for i in groups["val"]])
        val_f1, _, _ = evaluate(model, val_waves, val_labels, filters, window, mean, std, dev)
        print(f"epoch {epoch:02d} loss {total / seen:.3f} val_macro_f1 {val_f1:.3f}", flush=True)
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_idx = np.array(groups["test"])
    test_labels = np.array([label_index[manifest[i]["label"]] for i in test_idx])
    test_f1, pred, matrix = evaluate(
        model, audio[test_idx], test_labels, filters, window, mean, std, dev
    )
    balanced = float(np.mean([
        matrix[i, i] / matrix[i].sum() if matrix[i].sum() else 0.0 for i in range(4)
    ]))
    print(f"test macro F1 {test_f1:.3f} balanced accuracy {balanced:.3f}", flush=True)
    print(np.array(matrix), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "data" / "cache" / "encoder.pt"
    torch.save({"state": best_state, "labels": LABELS}, ckpt)

    model_cpu = VoiceEncoder().cpu().eval()
    model_cpu.load_state_dict(best_state)
    dummy = torch.zeros(1, n_frames(), N_MELS)
    onnx_path = OUT / "talksik.onnx"
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model_cpu(dummy).numpy()
    got = session.run(None, {"mel": dummy.numpy()})[0]
    delta = float(np.max(np.abs(ref - got)))
    print(f"onnx max abs diff {delta:.3e}", flush=True)
    if delta > 1e-4:
        raise SystemExit("ONNX export does not match PyTorch")

    # One real clinic vowel per class, for the in-browser demo button.
    examples = []
    for label in LABELS:
        hits = [i for i in test_idx if manifest[i]["label"] == label]
        if not hits:
            hits = [i for i in range(len(manifest)) if manifest[i]["label"] == label]
        index = int(hits[0])
        mel = batch_mel(audio[index : index + 1], filters, window, mean, std)[0]
        examples.append({"label": label, "mel": mel.astype(float).round(5).tolist()})

    card = {
        "labels": LABELS,
        "display": {
            "healthy": "Healthy",
            "hyperfunctional_dysphonia": "Hyperfunctional dysphonia",
            "vocal_fold_paralysis": "Vocal-fold paralysis",
            "laryngitis": "Laryngitis",
        },
        "sample_rate": 16000,
        "window": 400,
        "hop": 160,
        "n_fft": 512,
        "n_mels": N_MELS,
        "frames": n_frames(),
        "target_samples": 20000,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "filterbank": filters.astype(float).tolist(),
        "window_fn": window.astype(float).tolist(),
        "test_macro_f1": round(test_f1, 3),
        "test_balanced_accuracy": round(balanced, 3),
        "confusion": matrix.tolist(),
        "test_clips": int(len(test_idx)),
        "train_clips": int(len(groups["train"])),
        "examples": examples,
        "dataset": "Saarbrücken Voice Database",
        "dataset_url": "https://zenodo.org/records/16874898",
        "license": "CC BY 4.0",
    }
    hubert_path = CACHE / "hubert.json"
    if hubert_path.exists():
        card.update(json.loads(hubert_path.read_text()))
    (OUT / "meta.json").write_text(json.dumps(card))
    print(f"wrote {onnx_path} and {OUT / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
