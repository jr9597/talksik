# Talksik

Talksik fine-tunes HuBERT-base to tell four voice groups apart from someone talking: healthy, hyperfunctional dysphonia, vocal-fold paralysis, and laryngitis.

The recording that counts is one German sentence, “Guten Morgen, wie geht es Ihnen?” Speakers are held out, so the test score is on people the model has never heard. Sustained vowels from the same session are extra training clips with that speaker’s label.

This is a research model of publicly recorded voices. It does not diagnose anyone.

## Data

The audio is the [Saarbrücken Voice Database](https://zenodo.org/records/16874898) (Pützer and Barry, Zenodo record 16874898, CC BY 4.0). The four zips are **11.0 GB**. Each diagnosis is one zip, and the zip name is the label. Together they hold **1,199 sentences** and **16,279 vowels** (17,478 clips). After the speaker split that is 12,361 train clips, 2,601 validation clips, and 2,516 test clips. Repeating the training sentences brings the training signal to 23,372 examples.

| Zip | Label | Sentences | Vowels |
| --- | --- | ---: | ---: |
| `healthy.zip` | healthy | 634 | 8,931 |
| `Hyperfunktionelle Dysphonie.zip` | hyperfunctional dysphonia | 212 | 2,768 |
| `Rekurrensparese.zip` | vocal-fold paralysis | 213 | 2,760 |
| `Laryngitis.zip` | laryngitis | 140 | 1,820 |

Inside a speaker folder, the sentence is the `-phrase.nsp` file under `/sentences/`. Vowel clips are `.nsp` files under `/vowels/`. Recordings are 50 kHz. The script resamples them to 16 kHz, removes the DC offset, and caps each clip at 15 seconds. Electroglottography (`.egg`) files are ignored.

`overview.csv` supplies the speaker id. Speakers are split 70/15/15 inside each label, with seed 7, so no person appears in more than one split. Vowels outnumber sentences, so training repeats the sentences until they are about half of what the model sees. Validation and the headline test use the sentence only.

## Model

The starting point is [`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960). The fine-tuned network is **94.6 million parameters** (about 380 MB in 32-bit). The convolutional frontend, 4.2 million of those parameters, stays frozen. The other **90.4 million** are trained: a 12-layer transformer (hidden size 768, 12 heads, feed-forward 3,072), a learned weighted sum of its layers, a projection to 256, and a four-way classifier. The input is the raw waveform. Class-weighted cross-entropy balances the four groups.

## Where this is strong

The headline number is a person talking, with that person’s other recordings kept out of the test. A vowel-only score would answer a different question, and a clip-level split would let the same voice appear on both sides.

HuBERT-base is sized for the data. There are 1,199 labeled sentences. A model of this size, with the waveform frontend frozen and the transformer adapted, can learn from that without the capacity of a speech model in the billions. The vowels add more audio from the same speakers, and the repetition keeps the sentence at about half of training.

The evidence is this corpus and this one prompted sentence, recorded at 50 kHz in the Saarbrücken protocol. Healthy speakers are 634 of the 1,199 sentences. There is no second corpus here, and no test on phone audio or other languages.

Training uses a learning rate of 2e-5, a 10% warmup, weight decay 0.01, batches of 8 with gradient accumulation of 2, and 16-bit precision on a GPU. It runs for up to 12 epochs and stops early if validation sentence macro F1 does not improve for 3 epochs. Training clips get a small random gain and a little noise.

## Train

The fine-tune runs on a Hugging Face Jobs T4. Jobs need a paid Hugging Face plan with prepaid credits. A T4 is about $0.40 an hour, and a full run fits in an 8 hour cap.

```bash
uv run hf auth login
uv run hf jobs uv run \
    --flavor t4-small \
    --timeout 8h \
    --name talksik-hubert \
    --detach \
    --secrets HF_TOKEN \
    train/finetune.py
```

`train/finetune.py` is a self-contained script. The job downloads the four Zenodo zips, trains, and prints a line starting with `TEST`. When `HF_TOKEN` can write to the Hub, it pushes a private model to `your-username/talksik-hubert` along with `metrics.json`.

```bash
uv run hf jobs logs -f your-username/JOB_ID
```

The download log prints one line when each zip starts and stays there until that file finishes. `healthy.zip` is 6 GB, so that first line sits for a while.

## Browser sketch

`web/` is an earlier in-browser page that scores a sustained “ah” with a small ONNX encoder. The sentence model above is the one this project is training.
