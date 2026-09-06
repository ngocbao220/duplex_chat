# Integrated pipeline provenance

- Cholimex: retained from this repository's `cholimex` branch.
- DuplexChat single-audio runner: imported from local `master` commit
  `24019279ffff873b754a2f7f690397f14e2b3f94`, originally `test_single.py`.
  Production code now lives in `src/duplexchat_pipe/single_audio.py`; the old
  filename is a compatibility entrypoint. Import no longer patches `torch.load`
  globally. Device is resolved at invocation, temporary audio is sample-local,
  and output uses the repository WAV writer. The separation algorithm and
  default model/chunk settings remain those of the source runner.
- Vilier: imported tracked source from local sibling repo commit
  `980d9f83cb1ae25c8adbe91abd014828d2b19d9a`; see
  `pipelines/vilier/SOURCE.json`. Original source checkout was not modified.
  Source, tests, dependency profiles, scripts, notebooks and SepReformer code
  are included. No source Git directory, model weights, data, cache or results
  are included. Upstream model code licenses remain with the imported code.

## Integration adaptations

- The original Vilier project manifest is retained as `pyproject.upstream.toml`.
  Runtime projects are non-workspace uv projects with independent locks.
- ClearVoice and pyannote 4 have incompatible NumPy constraints. The comparison
  adapter supplies a persistent ClearVoice helper process in its own NumPy 1.x
  environment. Native Vilier algorithm code is retained; the adapter treats
  configured separation loader failures as sample failures instead of silent
  degradation. ASR and Qwen are disabled for comparison.
- Imported notebook outputs/execution counters were cleared. Native notebook
  workflows are reference material; the integrated Kaggle workflow is in the
  root `notebook/kaggle.ipynb`.
- Vilier tests now generate their 30-second waveform fixture. Two stale test
  expectations were corrected: restoring a previously imported pyannote module,
  and translating legacy `use_auth_token` to modern `token` rather than dropping
  authentication. No test is skipped.
- Root Kaggle tests now cover the notebook's integrated smoke/comparison flow
  and accept standard notebook source arrays. The prior tests expected removed
  model-sweep cells that were absent from the source notebook before integration.

- DialogueSidon cached `.pt2` metadata declares `torch_version=2.8.0+cu128`.
  Torch 2.14 failed deserialization during real smoke, so its two runtime
  projects pin Torch/TorchAudio 2.8 and compatible TorchCodec 0.7.
- CPU deserialization of DialogueSidon exports uses a scoped serializer adapter
  that maps CUDA-authored sample artifacts to CPU before moving each module to
  the selected device. The serializer is restored in `finally`; `torch.load`
  remains unchanged. Export loader stderr is no longer discarded, so checkpoint
  failures retain their diagnostic messages.
