# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate",
#   "numpy",
#   "scipy",
#   "torch",
#   "transformers",
# ]
# ///
"""Fine-tune HuBERT-base on Saarbrücken connected speech.

The person says a sentence. Vowels are extra clips with the same label.
The score that matters is the sentence test, split by speaker.
The convolutional frontend stays frozen.
"""

from __future__ import annotations

import csv
import io
import json
import os
import struct
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MODEL_ID = "facebook/hubert-base-ls960"
SEED = 7
SR = 16_000
LABELS = [
    "healthy",
    "hyperfunctional_dysphonia",
    "vocal_fold_paralysis",
    "laryngitis",
]
ZIPS = {
    "healthy.zip": ("healthy", "https://zenodo.org/api/records/16874898/files/healthy.zip/content", 6_020_018_616),
    "Hyperfunktionelle Dysphonie.zip": (
        "hyperfunctional_dysphonia",
        "https://zenodo.org/api/records/16874898/files/Hyperfunktionelle%20Dysphonie.zip/content",
        1_928_321_725,
    ),
    "Rekurrensparese.zip": (
        "vocal_fold_paralysis",
        "https://zenodo.org/api/records/16874898/files/Rekurrensparese.zip/content",
        1_900_446_365,
    ),
    "Laryngitis.zip": (
        "laryngitis",
        "https://zenodo.org/api/records/16874898/files/Laryngitis.zip/content",
        1_193_637_707,
    ),
}
MAX_SAMPLES = SR * 15


def read_nsp(data: bytes) -> tuple[np.ndarray, int]:
    if data[:4] != b"FORM":
        raise ValueError("not an NSP file")
    idx = data.find(b"SDA_")
    if idx < 12:
        raise ValueError("NSP data chunk missing")
    sample_rate = struct.unpack_from("<I", data, idx - 12)[0]
    nbytes = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = np.frombuffer(data[idx + 8 : idx + 8 + nbytes], dtype="<i2").copy()
    return pcm.astype(np.float32) / 32768.0, int(sample_rate)


def to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SR:
        return audio
    from scipy.signal import resample_poly

    g = int(np.gcd(src_sr, SR))
    return resample_poly(audio, SR // g, src_sr // g).astype(np.float32)


def download(url: str, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    while not dest.exists() or dest.stat().st_size < size:
        have = dest.stat().st_size if dest.exists() else 0
        print(f"download {dest.name} {have}/{size}", flush=True)
        headers = {"User-Agent": "talksik"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                code = getattr(response, "status", 200)
                mode = "ab" if code == 206 and have else "wb"
                with dest.open(mode) as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
        except Exception as exc:
            print(f"retry {dest.name}: {exc}", flush=True)
            time.sleep(5)
            continue
        if dest.exists() and dest.stat().st_size >= size:
            return
        time.sleep(3)


def load_clips(zip_dir: Path) -> list[dict]:
    clips = []
    for filename, (label, _url, _size) in ZIPS.items():
        path = zip_dir / filename
        print(f"reading {filename}", flush=True)
        with zipfile.ZipFile(path) as zf:
            meta = {}
            raw = zf.read("overview.csv").decode("utf-8-sig", errors="replace")
            for row in csv.DictReader(io.StringIO(raw)):
                recording = row.get("AufnahmeID", "").strip()
                if recording and recording not in meta:
                    meta[recording] = (
                        row.get("SprecherID", "").strip() or recording,
                        row.get("Geschlecht", "").strip().lower(),
                    )
            names = []
            for name in zf.namelist():
                if name.endswith("-phrase.nsp") and "/sentences/" in name:
                    names.append((name, "speech"))
                elif name.endswith(".nsp") and "/vowels/" in name:
                    names.append((name, "vowel"))
            n_speech = sum(task == "speech" for _name, task in names)
            print(f"  {n_speech} sentences, {len(names) - n_speech} vowels", flush=True)
            for name, task in names:
                recording = name.split("/", 1)[0]
                audio, src_sr = read_nsp(zf.read(name))
                audio = to_16k(audio, src_sr)
                if audio.size > MAX_SAMPLES:
                    audio = audio[:MAX_SAMPLES]
                audio = audio - float(audio.mean())
                speaker, sex = meta.get(recording, (recording, ""))
                clips.append({
                    "audio": np.ascontiguousarray(audio, dtype=np.float32),
                    "label": LABELS.index(label),
                    "speaker": speaker,
                    "sex": sex,
                    "label_name": label,
                    "task": task,
                })
    return clips


def by_task(clips: list[dict], indices: list[int], task: str) -> list[int]:
    return [i for i in indices if clips[i]["task"] == task]


def training_indices(clips: list[dict], indices: list[int]) -> list[int]:
    """Repeat sentences until they are about half the training signal."""
    speech = by_task(clips, indices, "speech")
    vowel = by_task(clips, indices, "vowel")
    if not speech:
        return list(indices)
    reps = max(1, round(len(vowel) / len(speech))) if vowel else 1
    return vowel + speech * reps


def split_indices(clips: list[dict]) -> dict[str, list[int]]:
    rng = np.random.default_rng(SEED)
    by_label: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: set[tuple[str, str]] = set()
    for clip in clips:
        key = (clip["label_name"], clip["speaker"])
        if key in seen:
            continue
        seen.add(key)
        by_label[clip["label_name"]].append(clip["speaker"])
    assignment: dict[str, str] = {}
    for label, speakers in by_label.items():
        speakers = np.array(speakers)
        rng.shuffle(speakers)
        n_test = max(1, round(len(speakers) * 0.15))
        n_val = max(1, round(len(speakers) * 0.15))
        for speaker in speakers[:n_test]:
            assignment[f"{label}:{speaker}"] = "test"
        for speaker in speakers[n_test : n_test + n_val]:
            assignment[f"{label}:{speaker}"] = "val"
        for speaker in speakers[n_test + n_val :]:
            assignment[f"{label}:{speaker}"] = "train"
    groups = {"train": [], "val": [], "test": []}
    for i, clip in enumerate(clips):
        groups[assignment[f"{clip['label_name']}:{clip['speaker']}"]].append(i)
    return groups


class Clips(Dataset):
    def __init__(self, clips: list[dict], indices: list[int], train: bool):
        self.clips = clips
        self.indices = indices
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        clip = self.clips[self.indices[i]]
        audio = clip["audio"]
        if self.train:
            rng = np.random.default_rng(SEED + i + len(audio))
            audio = audio * np.float32(10 ** (rng.uniform(-6, 3) / 20))
            audio = audio + rng.normal(0, 0.003, size=audio.shape).astype(np.float32)
        return {"audio": np.asarray(audio, dtype=np.float32), "label": clip["label"]}


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float, list[list[int]]]:
    n = len(LABELS)
    matrix = np.zeros((n, n), dtype=int)
    for g, p in zip(gold, pred):
        matrix[int(g), int(p)] += 1
    scores = []
    recalls = []
    for i in range(n):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        recalls.append(rec)
    return float(np.mean(scores)), float(np.mean(recalls)), matrix.tolist()


def main() -> None:
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        EarlyStoppingCallback,
        HubertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(SEED)
    zip_dir = Path(os.environ.get("TALKS_ZIPS", "/tmp/talksik-zips"))
    out_dir = Path(os.environ.get("TALKS_OUT", "/tmp/talksik-hubert"))
    for filename, (_label, url, size) in ZIPS.items():
        download(url, zip_dir / filename, size)

    clips = load_clips(zip_dir)
    groups = split_indices(clips)
    train_idx = training_indices(clips, groups["train"])
    val_speech = by_task(clips, groups["val"], "speech") or groups["val"]
    test_speech = by_task(clips, groups["test"], "speech")
    test_vowel = by_task(clips, groups["test"], "vowel")
    print(
        "clips",
        {name: len(idx) for name, idx in groups.items()},
        "train_with_repeated_sentences",
        len(train_idx),
        flush=True,
    )
    counts = np.bincount([clips[i]["label"] for i in groups["train"]], minlength=len(LABELS))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    weights = weights / weights.mean()

    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        use_weighted_layer_sum=True,
        classifier_proj_size=256,
    )
    # The weighted sum is sized for every layer. Pretraining layerdrop skips
    # layers at random and then the sum crashes.
    config.layerdrop = 0.0
    model = HubertForSequenceClassification.from_pretrained(MODEL_ID, config=config)
    model.freeze_feature_encoder()

    def collate(batch: list[dict]) -> dict:
        encoded = processor(
            [item["audio"] for item in batch],
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return encoded

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        f1, balanced, _matrix = macro_f1(np.argmax(logits, axis=-1), gold)
        return {"macro_f1": f1, "balanced_accuracy": balanced}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=0.1,
        weight_decay=0.01,
        num_train_epochs=12,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=Clips(clips, train_idx, train=True),
        eval_dataset=Clips(clips, val_speech, train=False),
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    def score(indices: list[int]) -> tuple[float, float, list[list[int]]]:
        prediction = trainer.predict(Clips(clips, indices, train=False))
        return macro_f1(np.argmax(prediction.predictions, axis=-1), prediction.label_ids)

    f1, balanced, matrix = score(test_speech)
    vowel_f1, vowel_balanced, vowel_matrix = score(test_vowel) if test_vowel else (None, None, None)
    report = {
        "model": MODEL_ID,
        "task": "sentence",
        "test_macro_f1": round(f1, 4),
        "test_balanced_accuracy": round(balanced, 4),
        "confusion": matrix,
        "vowel_test_macro_f1": None if vowel_f1 is None else round(vowel_f1, 4),
        "vowel_test_balanced_accuracy": None if vowel_balanced is None else round(vowel_balanced, 4),
        "vowel_confusion": vowel_matrix,
        "labels": LABELS,
        "test_sentences": len(test_speech),
        "test_vowels": len(test_vowel),
        "train_clips": len(groups["train"]),
        "val_sentences": len(val_speech),
        "split": "speaker",
    }
    print("TEST " + json.dumps(report), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        from huggingface_hub import whoami

        username = whoami(token=token)["name"]
        repo = f"{username}/talksik-hubert"
        model.push_to_hub(repo, private=True, token=token)
        processor.push_to_hub(repo, private=True, token=token)
        from huggingface_hub import HfApi

        HfApi(token=token).upload_file(
            path_or_fileobj=str(out_dir / "metrics.json"),
            path_in_repo="metrics.json",
            repo_id=repo,
            repo_type="model",
        )
        print(f"pushed https://huggingface.co/{repo}", flush=True)


if __name__ == "__main__":
    main()
