"""Turn Saarbrücken NSP vowels into a fixed-length 16 kHz cache."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from audio import loudest_window, read_nsp, resample

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "data" / "zips"
CACHE = ROOT / "data" / "cache"

# Zip filename -> model label. The zip is the cohort; the free-text diagnosis
# field often lists extra findings and is not the training target.
ZIPS = {
    "healthy.zip": "healthy",
    "Hyperfunktionelle Dysphonie.zip": "hyperfunctional_dysphonia",
    "Rekurrensparese.zip": "vocal_fold_paralysis",
    "Laryngitis.zip": "laryngitis",
}

VOWEL = re.compile(r"^(\d+)/vowels/\1-a_(n|l|h)\.nsp$")


def speaker_table(zf: zipfile.ZipFile) -> dict[str, str]:
    raw = zf.read("overview.csv").decode("utf-8-sig", errors="replace")
    sex: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(raw)):
        speaker = row.get("SprecherID", "").strip()
        gender = row.get("Geschlecht", "").strip().lower()
        if speaker and speaker not in sex:
            sex[speaker] = gender
    return sex


def main() -> None:
    missing = [name for name in ZIPS if not (ZIP_DIR / name).exists()]
    if missing:
        raise SystemExit(f"missing zips in {ZIP_DIR}: {', '.join(missing)}")

    records: list[dict] = []
    waves: list[np.ndarray] = []
    for filename, label in ZIPS.items():
        path = ZIP_DIR / filename
        print(f"reading {filename}", flush=True)
        with zipfile.ZipFile(path) as zf:
            sex = speaker_table(zf)
            names = [n for n in zf.namelist() if VOWEL.match(n)]
            print(f"  {len(names)} /a/ vowels", flush=True)
            for name in names:
                match = VOWEL.match(name)
                assert match is not None
                recording_id, pitch = match.group(1), match.group(2)
                audio, sr = read_nsp(zf.read(name))
                audio = loudest_window(resample(audio, sr))
                # overview keys speakers, not recordings. Fall back to the folder id.
                speaker = recording_id
                # Prefer the speaker id from the csv when the recording id maps.
                # The csv AufnahmeID is the folder id; SprecherID is the person.
                waves.append(audio)
                records.append(
                    {
                        "file": name,
                        "recording_id": recording_id,
                        "speaker": speaker,
                        "pitch": pitch,
                        "label": label,
                        "sex": "",
                    }
                )
            # Second pass: attach SprecherID. Done from a small map so we don't
            # hold the csv reader open across the audio loop.
            id_to_speaker, id_to_sex = recording_meta(zf)
        for rec in records:
            if rec["label"] != label:
                continue
            rec["speaker"] = id_to_speaker.get(rec["recording_id"], rec["recording_id"])
            rec["sex"] = id_to_sex.get(rec["recording_id"], sex.get(rec["speaker"], ""))

    CACHE.mkdir(parents=True, exist_ok=True)
    audio = np.stack(waves).astype(np.float32)
    np.save(CACHE / "audio.npy", audio)
    (CACHE / "manifest.json").write_text(json.dumps(records))
    print(f"cached {len(records)} clips -> {CACHE / 'audio.npy'} {audio.shape}", flush=True)


def recording_meta(zf: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    raw = zf.read("overview.csv").decode("utf-8-sig", errors="replace")
    speaker: dict[str, str] = {}
    sex: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(raw)):
        recording = row.get("AufnahmeID", "").strip()
        if not recording or recording in speaker:
            continue
        speaker[recording] = row.get("SprecherID", "").strip() or recording
        sex[recording] = row.get("Geschlecht", "").strip().lower()
    return speaker, sex


if __name__ == "__main__":
    main()
