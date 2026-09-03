# DuplexChat

**DuplexChat** is a large-scale, bilingual (Japanese + English), two-speaker
**full-duplex spoken-dialogue** corpus built from public podcast feeds. Each
clip is a natural two-person conversation, separated into one speaker per
channel (stereo: L = speaker A, R = speaker B), suitable for training and
evaluating full-duplex dialogue models.

To stay copyright-safe, the dataset is **distributed as metadata only** — a
manifest of pointers to the original public source audio plus the dialogue-span
timings — from the HuggingFace dataset
[`sarulab-speech/DuplexChat`](https://huggingface.co/datasets/sarulab-speech/DuplexChat).
No audio is redistributed here.

This repository contains the code for two things:

- **Part A — Reconstruct the dataset** from the released metadata
  (`scripts/reconstruct_dataset.py`).
- **Part B — Re-run the full construction pipeline** from scratch
  (`duplexchat-pipe`), including producing the metadata manifest itself.

Both share one installable package, `duplexchat-pipe` (import name
`duplexchat_pipe`).

## Requirements

- Python 3.12+
- `ffmpeg` + `ffprobe` on `PATH`
- A CUDA GPU (for diarization and DialogueSidon separation)
- A HuggingFace token with access to the gated model weights
  ([`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
  used only for crawling, and
  [`sarulab-speech/DialogueSidon`](https://huggingface.co/sarulab-speech/DialogueSidon),
  used for separation):

  ```bash
  huggingface-cli login
  ```

## Install

```bash
uv sync
```

`torch` / `torchaudio` / `torchcodec` resolve from PyPI by default so `uv sync`
works on any machine.

### Selecting a CUDA build

To pin a specific CUDA wheel (recommended on GPU clusters), add a PyTorch wheel
index to `pyproject.toml` and route the three packages to it — a ready-to-paste
block is in `pyproject.toml`. For example, for CUDA 12.4 use
`https://download.pytorch.org/whl/cu124`; adjust the URL for your CUDA version
(`cu121`, `cu124`, `cu130`, …). Then re-run `uv sync`.

---

## Part A — Reconstruct the dataset

1. Download the manifest for the language you want from the HuggingFace dataset,
   e.g. `duplexchat_manifest_en.jsonl.gz` (English) or
   `duplexchat_manifest_ja.jsonl.gz` (Japanese).

2. Reconstruct:

   ```bash
   uv run python3 scripts/reconstruct_dataset.py \
       --manifest duplexchat_manifest_en.jsonl.gz \
       --output ./data/reconstructed_en
   ```

For each dialogue clip the script downloads the source episode (once per
episode), transcodes it to 16 kHz mono, slices the dialogue span, runs
DialogueSidon two-speaker separation, and writes a stereo MP3 into WebDataset
`.tar.gz` shards that match the layout of the original dataset. The result is
byte-faithful to the released corpus, because reconstruction reuses the same
pipeline code.

**Distributed reconstruction.** Shard the work across many GPUs by stable hash
of the episode URL (so every dialogue of an episode is downloaded once, on one
worker):

```bash
uv run python3 scripts/reconstruct_dataset.py \
    --language en --manifest-dir ./data/manifest \
    --output ./data/reconstructed_en \
    --num-shards 64 --shard-index $I     # I = 0 .. 63, one per worker
```

Runs are **resumable**: completed clips are recorded in a small SQLite ledger
under `<output>/cache/`, so re-running skips finished work. Episodes whose
source URL is dead are logged and skipped (expected as source feeds age).
See `jobs/reconstruct.sh` for a cluster array-job template.

Notes:
- Requires the gated `sarulab-speech/DialogueSidon` weights and a GPU.
- Per-clip diarization segments are **not** part of the released metadata, so a
  reconstructed sample contains `audio.mp3` (the separated stereo clip) and
  `meta.json` (rebuilt from the manifest row) — no `diarization.json`.

### Manifest row schema

Each line of a manifest is one two-speaker dialogue clip:

```json
{
  "language": "en-us",
  "rss_url": "https://.../podcast/rss",
  "audio_url": "https://.../episode.mp3",
  "dialogue_idx": 1,
  "episode_start_sec": 80.22,
  "episode_end_sec": 708.44,
  "duration_sec": 628.22,
  "speakers": ["SPEAKER_04", "SPEAKER_02"],
  "episode_duration_sec": 2165.52
}
```

Rows are globally deduplicated by `(audio_url, dialogue_idx)`.

---

## Part B — Re-run the pipeline

The construction pipeline crawls podcast feeds, extracts two-speaker dialogues,
separates them, and writes WebDataset shards.

By default, `duplexchat-pipe run` reads `config.json`. Use dotted `key=value`
overrides for one-off runs.

```bash
# Whole-episode audio only (no diarization):
uv run duplexchat-pipe run --output ./data/wds \
    diarization.enabled=false separation.enabled=false

# + diarization (required for dialogue extraction):
uv run duplexchat-pipe run --output ./data/wds \
    diarization.enabled=true separation.enabled=false

# + DialogueSidon two-speaker separation (stereo output):
uv run duplexchat-pipe run --output ./data/wds \
    diarization.enabled=true separation.enabled=true
```

Supported diarization model aliases:

- `sortformer` -> `nvidia/diar_sortformer_4spk-v1` (`--diarization-backend sortformer`)
- `pyannote-3.1` -> `pyannote/speaker-diarization-3.1` (`--diarization-backend pyannote`)
- `diarizen` -> `BUT-FIT/diarizen-wavlm-large-s80-md` (`--diarization-backend diarizen`)

Supported two-speaker separation backends:

- `dialoguesidon` -> `sarulab-speech/DialogueSidon`
- `sepformer` -> `speechbrain/sepformer-wsj02mix`
- `mossformer2` -> `alibabasglab/MossFormer2_SS_16K`

Examples:

```bash
uv run duplexchat-pipe run --enable-diarization \
    --diarization-backend sortformer --diarization-model sortformer

uv run duplexchat-pipe run --enable-diarization --enable-separation \
    --separation-backend sepformer --separation-model sepformer
```

Backend dependency stacks can conflict, so install only the profiles you need:

```bash
uv sync --extra diarization-pyannote --extra separation-dialoguesidon
uv sync --extra diarization-sortformer
uv sync --extra separation-sepformer
uv pip install -r requirements/diarization-diarizen.txt
uv pip install -r requirements/separation-mossformer2.txt
```

For Sortformer, `requirements/diarization-sortformer.txt` follows NVIDIA's
NeMo-from-GitHub install path when the PyPI NeMo package is not sufficient.

### Architecture

Four concurrent thread pools form a producer–consumer chain:

1. **RSS pool** (`--rss-workers`) — parses the podcastindex.org feed DB and RSS
   feeds, yielding audio items.
2. **Download pool** (`--download-workers`) — downloads raw audio, probes
   duration via `ffprobe`.
3. **Diarize pool** (`--process-workers`) — transcodes to 16 kHz mono, runs
   pyannote diarization, extracts 2-speaker dialogue runs (default: turns split
   on ≥5 s gaps; each dialogue ≥10 s and ≤10 min; no single speaker >80% of turn
   time; episodes with <4 valid dialogues are dropped).
4. **Separation pool** (`--separation-workers`) — runs DialogueSidon on each
   dialogue and emits a stereo MP3 sample.

### Output schema (dialogue-level)

With `--enable-diarization --enable-separation`, each sample is one dialogue:

- `__key__`: `<sha1(normalized audio_url)>_<dialogue_idx:04d>`
- `audio.mp3`: stereo MP3 — channel 0 = speaker A, channel 1 = speaker B
- `meta.json`: rss/audio URLs, language, episode + dialogue timings, speaker
  labels, separation model info
- `diarization.json`: the episode's diarization segments

### Debug phase outputs

Runs also write inspectable phase artifacts under `outputs/` by default. For
each processed episode or single-audio model test, diarization labels are in:

```text
outputs/<episode_or_model>/phase_02_diarization/labels/
```

The label files are Audacity-compatible `start<TAB>end<TAB>Label` text files:
`speakers.txt`, `vad.txt`, and one file per normalized speaker such as
`SPEAKER_00.txt` and `SPEAKER_01.txt`. Disable these artifacts for large crawls
with `--no-debug-outputs`, or redirect them with `--debug-outputs-dir`.

### Distributed / resumable crawling

Feeds are sharded across nodes by stride `feed_urls[node_index::num_nodes]`.
Pass `--node-index N --num-nodes M`; each node keeps its own `processed.sqlite`
under `<cache-dir>/<node_index>/` so re-runs skip finished episodes.
**`--num-nodes` must stay constant across re-runs** or the stride assignment
(and thus dedup) breaks. See `jobs/crawl.sh`.

### Vietnamese POC with config.json

The Vietnamese proof-of-concept config lives in `config.json`. Bash scripts read
that file and forward `key=value` overrides.

```bash
# End-to-end Vietnamese POC
bash jobs/vi/run_end2end.sh source.target_hours=10

# Use MossFormer2 instead of DialogueSidon
bash jobs/vi/run_end2end.sh separation=mossformer2

# Use Sortformer or DiariZen diarization
bash jobs/vi/run_end2end.sh diarization=sortformer
bash jobs/vi/run_end2end.sh diarization=diarizen

# Use SepFormer instead of DialogueSidon
bash jobs/vi/run_end2end.sh separation=sepformer

# Auto device resolution is cuda -> cpu. Multi-GPU is opt-in:
bash jobs/vi/run_end2end.sh \
    runtime.device=auto \
    runtime.multi_gpu.enabled=true \
    runtime.multi_gpu.device_ids='[0,1]'
```

Logs are written to `logs/YYYY-MM-DD/<run_id>/` with `run.log`,
`resolved_config.yaml`, `phase_tree.txt`, `stats_table.md`, `errors.jsonl`, and
`artifacts.json`. The same config can run individual phases through
`jobs/vi/01_collect_sources.sh` ... `jobs/vi/06_benchmark.sh`.

Equivalent direct CLI usage:

```bash
uv run duplexchat-pipe run --phase end2end source.target_hours=2.5
uv run duplexchat-pipe run --config config.json --phase benchmark
```

### Post-processing: filter + dedup

```bash
uv run python3 scripts/filter_shards.py \
    --input data/wds_en --output data/wds_en_filtered \
    --shard-size-gb 3.0 --num-workers 8
```

Drops music-genre feeds, `(audio_url, dialogue_idx)` duplicates, and dialogues
that are ≥1/3 of the parent episode (usually diarization failures on monologue
content), and writes a `stats.json` report. See `jobs/filter.sh`.

### Export the reconstruction metadata

Regenerate the copyright-safe manifest from built shards:

```bash
# Array task i of N scans shards[i::N]:
uv run python3 analysis/export_manifest_scan.py \
    --data-root ./data --task-id $I --num-tasks 16 --out-dir ./data/manifest_parts
# Then merge + globally dedup:
uv run python3 analysis/export_manifest_reduce.py \
    --parts-dir ./data/manifest_parts --out-dir ./data/manifest
```

See `jobs/export_manifest.sh`.

## Repository layout

```
src/duplexchat_pipe/          # the construction pipeline package
scripts/reconstruct_dataset.py# Part A — rebuild audio from the manifest
scripts/filter_shards.py      # post-crawl filter / dedup / stats
scripts/repair_shards.py      # repair truncated shards from killed jobs
analysis/export_manifest_*.py # produce the released metadata manifest
jobs/                         # cluster job templates (PBS + MPI; adapt to your site)
tests/                        # unit tests (uv run pytest)
```

## Tests

```bash
uv run pytest
```

## License and source-audio rights

The code in this repository and the DuplexChat reconstruction **metadata** are
released under the **MIT License** (see [LICENSE](LICENSE)).

This license covers **only** the software and the metadata. It grants **no**
rights in the underlying source audio: copyright in the original podcast audio
remains with the respective publishers, and per-show terms vary. This project
redistributes no audio — audio is reconstructed locally by each user from the
public source URLs, and users are responsible for ensuring their use complies
with applicable terms and law.
