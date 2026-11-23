"""Frozen HuBERT-base embeddings, then a linear head. The browser model copies this head."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import LABELS, confusion, device, macro_f1, split_speakers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
MODEL_ID = "facebook/hubert-base-ls960"


def embed(audio: np.ndarray, dev: torch.device) -> np.ndarray:
    from transformers import AutoFeatureExtractor, HubertModel

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = HubertModel.from_pretrained(MODEL_ID).eval().to(dev)
    out = np.empty((len(audio), model.config.hidden_size), dtype=np.float32)
    batch = 8
    with torch.no_grad():
        for start in range(0, len(audio), batch):
            waves = [row for row in audio[start : start + batch]]
            inputs = extractor(waves, sampling_rate=16000, return_tensors="pt", padding=True)
            hidden = model(inputs.input_values.to(dev)).last_hidden_state
            pooled = hidden.mean(dim=1).cpu().numpy()
            out[start : start + len(waves)] = pooled
            if start % 400 == 0:
                print(f"embedded {start + len(waves)}/{len(audio)}", flush=True)
    return out


def fit_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dev: torch.device,
) -> tuple[nn.Linear, float]:
    head = nn.Linear(embeddings.shape[1], len(LABELS)).to(dev)
    counts = np.bincount(labels[train_idx], minlength=len(LABELS))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=dev)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    x_train = torch.from_numpy(embeddings[train_idx]).to(dev)
    y_train = torch.from_numpy(labels[train_idx]).to(dev)
    x_val = torch.from_numpy(embeddings[val_idx]).to(dev)
    y_val = labels[val_idx]
    best_f1 = -1.0
    best_state = None
    patience = 0
    for epoch in range(1, 41):
        head.train()
        order = torch.randperm(len(train_idx), device=dev)
        for start in range(0, len(order), 256):
            pick = order[start : start + 256]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(x_train[pick]), y_train[pick])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(x_val).argmax(dim=1).cpu().numpy()
        matrix = np.array(confusion(pred, y_val, len(LABELS)))
        score = macro_f1(matrix)
        print(f"hubert head epoch {epoch:02d} val_macro_f1 {score:.3f}", flush=True)
        if score > best_f1 + 1e-4:
            best_f1 = score
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    return head, best_f1


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text())
    audio = np.load(CACHE / "audio.npy")
    assignment = split_speakers(manifest)
    label_index = {label: i for i, label in enumerate(LABELS)}
    labels = np.array([label_index[row["label"]] for row in manifest])
    groups = {"train": [], "val": [], "test": []}
    for i, row in enumerate(manifest):
        groups[assignment[f"{row['label']}:{row['speaker']}"]].append(i)
    dev = device()
    print(f"HuBERT probe on {dev}", flush=True)
    embeddings = embed(audio, dev)
    np.save(CACHE / "hubert_embeddings.npy", embeddings)
    head, _ = fit_head(
        embeddings,
        labels,
        np.array(groups["train"]),
        np.array(groups["val"]),
        dev,
    )
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(embeddings).to(dev)).cpu().numpy()
    np.save(CACHE / "teacher_logits.npy", logits.astype(np.float32))
    test_idx = np.array(groups["test"])
    pred = logits[test_idx].argmax(axis=1)
    matrix = np.array(confusion(pred, labels[test_idx], len(LABELS)))
    score = macro_f1(matrix)
    balanced = float(np.mean([
        matrix[i, i] / matrix[i].sum() if matrix[i].sum() else 0.0 for i in range(4)
    ]))
    print(f"hubert test macro F1 {score:.3f} balanced accuracy {balanced:.3f}", flush=True)
    (CACHE / "hubert.json").write_text(json.dumps({
        "hubert_model": MODEL_ID,
        "hubert_test_macro_f1": round(score, 3),
        "hubert_test_balanced_accuracy": round(balanced, 3),
        "hubert_confusion": matrix.tolist(),
    }))


if __name__ == "__main__":
    main()
