# DITS — Drums In the Trash Stratum 

Anonymous code release accompanying the ISMIR submission, including both DITS data synthesis and DITS model inference implementation.

1. **`transcribe.py`** — runs a pre-trained CRNN model (trained on the DITS dataset) on a drum audio file and returns Bass Drum (BD) / Snare Drum (SD) / Hi-Hats (HH) onset times.
2. **`render_loop.py`** — synthesizes a 16-step drum loop including Bass Drum, Snare Drum and Hihats, from configurable
   parameters (per-step velocities, tempo, swing, one-shot samples).

Both scripts share no global state with each other. You can run either
independently.

```
DITS/
├── README.md
├── requirements.txt
├── transcribe.py              # onset detection
├── render_loop.py             # drum-loop synthesis
├── adtof_pytorch/             # ADT model scripts
├── drum_machine/              # 16-step sequencer + monophonic renderer
├── checkpoints/               # ADT model checkpoints
└── examples/
    ├── one_shots/             # kick (BD) / snare (SD) / hihat (HH) one-shots used in demos
    ├── config_basic_house.json # Drum machine rhythmparameter examples
    └── config_swung_breakbeat.json
```

For the CRNN model implementation, we adapted the architecture and code from [`ADTOF_pytorch`](https://github.com/xavriley/ADTOF-pytorch), which is a Pytorch-based re-implementation of [ADTOF](https://github.com/mzehren/adtof).

---

## Installation

We recommend a fresh Python 3.10 or 3.11 environment.

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA, replace the `torch` line in `requirements.txt` with a CUDA wheel
(see comment at top of that file) and re-run `pip install -r requirements.txt`.

---

## 1. Onset transcription (`transcribe.py`)

Place the trained 3-channel checkpoint at `checkpoints/adtof_3ch.ckpt`
(any path works; the example below uses that location).

```bash
python transcribe.py path/to/drum_audio.wav \
    --checkpoint checkpoints/adtof_3ch.ckpt \
    --output onsets.json
```

This prints a per-instrument summary and writes `onsets.json`:

```json
{
  "kick":  [0.41, 0.91, 1.41, 1.92, ...],
  "snare": [0.92, 1.93, 2.93, ...],
  "hihat": [0.16, 0.41, 0.66, 0.91, ...]
}
```

### Programmatic use

```python
from transcribe import transcribe

onsets = transcribe(
    "path/to/drum_audio.wav",
    checkpoint_path="checkpoints/adt_model.ckpt",
    device="cuda",            # or "cpu"
    thresholds=(0.5, 0.5, 0.5),  # per-channel kick/snare/hihat
)
print(onsets["kick"])         # numpy array of seconds
```

### Checkpoint format

The loader accepts:
- `.ckpt` files saved by PyTorch Lightning during fine-tuning (keys prefixed
  with `model.`; the prefix is stripped automatically).
- Bare `.pth` state dicts whose keys match the `ADTOFFrameRNN` model.

The model is constructed with `output_classes=3`, so it expects checkpoints
trained with `three_ch=True`. The packaged 5-channel original ADTOF weights
in `adtof_pytorch/data/` are **not** compatible with the 3-channel head and
are kept only as a reference for the original model architecture.

---

## 2. Drum loop synthesis (`render_loop.py`)

Renders a 4-second 16-step monophonic drum loop. Configurable parameters:

- **One-shot samples** for kick / snare / hihat (any wav at any sample rate).
- **Per-step velocity vectors** (16 floats each in `[0, 1]`) — `0` = silence.
- **Tempo** in BPM.
- **Beat type**: `"8th"` or `"16th"` — determines which steps are
  swing-affected.
- **Per-track swing amount** in `[0.5, 0.75]` — `0.5` = no swing.
- **Output sample rate** and **loop duration** in seconds.
- **Render mode** (`--mode`):
  - `1` (default) — basic monophonic mix; divide by max-abs if any sample
    exceeds 1.0. No FX, no LUFS normalization. Output may have low loudness
    if the dry sum is well under unity.
  - `4` — RMS-aware pre-gain → per-track FX (HP filter, peak EQ, compressor,
    optional reverb on snare/hihat) → master bus (compressor + limiter at
    -1 dBFS) → LUFS normalize to `--target-lufs` (default -16) → clamp.
    FX parameters are sampled from instrument-specific random ranges; pass
    `--seed N` for reproducible renders.

### Render from a JSON config

```bash
# Mode 1: basic mix
python render_loop.py --config examples/config_basic_house.json \
    --output out/house_mode1.wav --save-stems

# Mode 4: with FX + LUFS normalization (reproducible via --seed)
python render_loop.py --config examples/config_basic_house.json \
    --mode 4 --seed 42 --output out/house_mode4.wav \
    --save-stems --save-log
```

`--save-log` writes a sibling JSON next to the output wav listing every
randomized FX parameter (EQ centre/gain/Q, compressor times, reverb
levels, RMS pre-gain, master gains, LUFS norm gain). Replaying with the
same `--seed` reproduces the same render bit-for-bit on the same machine.

`examples/config_basic_house.json`:

```json
{
    "kick_path":  "examples/one_shots/kick/kick.wav",
    "snare_path": "examples/one_shots/snare/snare.wav",
    "hihat_path": "examples/one_shots/hihat/hh.wav",
    "tempo": 124,
    "beat_type": "8th",
    "kick_velo":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "snare_velo": [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
    "hihat_velo": [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0],
    "swing_kick":  0.50,
    "swing_snare": 0.50,
    "swing_hihat": 0.55,
    "sample_rate": 16000,
    "loop_length_secs": 4.0,
    "one_shot_length_secs": 0.8
}
```

A second example (`examples/config_swung_breakbeat.json`) shows a 16th-note
breakbeat with hihat swing.

### Render from CLI flags

```bash
python render_loop.py \
    --kick  examples/one_shots/kick/kick.wav \
    --snare examples/one_shots/snare/snare.wav \
    --hihat examples/one_shots/hihat/hh.wav \
    --tempo 120 --beat-type 8th \
    --kick-velo  1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 \
    --snare-velo 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 \
    --hihat-velo 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 \
    --output out/cli.wav
```

### Programmatic use

```python
from drum_machine import render_loop

sum_track, tracks = render_loop(
    kick_path="examples/one_shots/kick/kick.wav",
    snare_path="examples/one_shots/snare/snare.wav",
    hihat_path="examples/one_shots/hihat/hh.wav",
    velo_kick=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    velo_snare=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    velo_hihat=[0.6] * 16,
    tempo=120.0,
    beat_type="8th",
    swing_hihat=0.55,
    sample_rate=44100,
)
# sum_track: (1, 176400) torch.Tensor; tracks: (3, 1, 176400) torch.Tensor
```

---

## End-to-end example

Render a loop, then transcribe it back to step times:

```bash
python render_loop.py --config examples/config_basic_house.json \
    --output out/house.wav

python transcribe.py out/house.wav \
    --checkpoint checkpoints/adtof_3ch.ckpt \
    --output out/house_onsets.json
```




---
## Supplementary_Materials

### Exploratory pilot study of evaluating ADTOF on 10 techno / house loops


### Parameters for audio effects chain used in generating DITS drum loops
shown in ```drum_machine/audio_processing.py```
