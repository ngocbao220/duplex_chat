from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from .contract import fingerprint, reusable, sha256, write_json
from .process import stream_process
from .report import comparison_report

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ('cholimex', 'duplexchat', 'vilier')


def phase(title):
    print(f'=====================Phase {title}', flush=True)


def code_identity(name):
    digest = hashlib.sha256()
    locations = [ROOT / 'comparison', ROOT / 'pipelines' / name]
    if name != 'vilier':
        locations.append(ROOT / 'src')
    for location in locations:
        for directory, subdirs, files in os.walk(location):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith('.') and d not in {'__pycache__', 'outputs', 'logs', 'inputs'})
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix in {'.py', '.toml', '.lock', '.yaml'}:
                    digest.update(str(path.relative_to(ROOT)).encode())
                    digest.update(path.read_bytes())
    return digest.hexdigest()


def safe_key(key):
    path = Path(key)
    if not key or path.is_absolute() or '..' in path.parts or any(part.startswith('.') for part in path.parts):
        raise ValueError(f'Invalid dataset sample key: {key!r}')
    return key


def prepare_samples(args, benchmark, save_wav):
    phase('1: Downloading OtoSpeech')
    local_dir = args.otospeech_root or benchmark.download_otospeech_dataset(
        repo_id=args.otospeech_repo, local_dir=args.otospeech_local_dir, max_download_gb=args.max_gb)
    samples = benchmark.discover_otospeech_samples(local_dir)
    if args.max_samples:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError('No OtoSpeech samples found in the selected subset')
    keys = [safe_key(sample['key']) for sample in samples]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate OtoSpeech sample keys')
    mixture_root = (args.mixture_root or args.output_root / 'otospeech_mixtures').resolve()
    phase('2: Mixing speaker streams')
    prepared = []
    for index, sample in enumerate(samples, 1):
        sample = dict(sample)
        try:
            mixture, _, rate = benchmark.mix_ground_truth_pair(
                Path(sample['gt_speaker_1']), Path(sample['gt_speaker_2']), args.sample_rate)
            if args.max_seconds:
                mixture = mixture[..., :max(1, round(args.max_seconds * rate))]
                sample['duration_limit_sec'] = mixture.shape[-1] / rate
            destination = mixture_root / sample['key']
            mixture_path = destination / 'mixture.wav'
            save_wav(mixture_path, mixture, rate)
            for field in ('gt_speaker_1', 'gt_speaker_2'):
                source = Path(sample[field]).resolve()
                target = destination / f'{field}.wav'
                if not target.exists() or not source.samefile(target):
                    shutil.copy2(source, target)
                sample[field] = str(target)
            sample.update(mixture=str(mixture_path), mixture_sha256=sha256(mixture_path))
        except Exception as exc:
            sample['preparation_error'] = f'{type(exc).__name__}: {exc}'
            print(f"[{index}/{len(samples)}] preparation failed {sample['key']}: {exc}", flush=True)
        prepared.append(sample)
        if not sample.get("preparation_error"):
            print(f"[{index}/{len(samples)}] mixed {sample['key']} -> {destination} (mixture + two originals)", flush=True)
    return prepared


def pipeline_config(name, args, cfg):
    if name == 'cholimex':
        return json.loads(json.dumps(asdict(cfg), default=str))
    path = args.vilier_config if name == 'vilier' else args.duplexchat_config
    config = json.loads(path.read_text())
    if name == 'vilier':
        config.setdefault('asr', {})['enabled'] = False
        config.setdefault('state_labeling', {})['enabled'] = False
        config.setdefault('runtime', {})['dry_run'] = False
        config.setdefault('entrypoint', {})['sample_rate'] = args.sample_rate
    return config


def launch_pipeline(name, request, run_dir):
    request_path = run_dir / name / 'request.json'
    write_json(request_path, request)
    project = ROOT / 'pipelines' / name
    env = dict(os.environ)
    # Never let an activated parent venv or uv project override collapse the isolated environments.
    env.pop('VIRTUAL_ENV', None)
    env['UV_PROJECT_ENVIRONMENT'] = str(project / '.venv')
    env['UV_CACHE_DIR'] = str(ROOT / '.uv-cache')
    env.pop('PYTHONPATH', None)
    command = ['uv', 'run', '--project', str(project), '--locked', '--no-dev',
               'python', str(ROOT / 'comparison/worker.py'), '--request', str(request_path)]
    try:
        return stream_process(command, project, run_dir / name / 'worker.log', env)
    except OSError as exc:
        print(f'[{name}] Cannot start worker: {exc}', flush=True)
        write_json(run_dir / name / 'launch_error.json', {'error': str(exc)})
        return 1


def run_otospeech(cfg, args, benchmark, save_wav):
    names = PIPELINES if args.pipeline == 'all' else (args.pipeline,)
    # Validate requested config files before downloading audio.
    configs = {name: pipeline_config(name, args, cfg) for name in names}
    prepared = prepare_samples(args, benchmark, save_wav)
    run_dir = (args.output_root / 'runs' / uuid.uuid4().hex).resolve()
    write_json(run_dir / 'samples.json', prepared)
    print(f'Shared dataset manifest: {run_dir / "samples.json"}', flush=True)
    pred_root = (args.pred_root or args.output_root / 'otospeech').resolve()
    report_base = (args.benchmark_output or cfg.benchmark_output_dir / 'summary.json').resolve()
    runtime, scored = {}, {}
    failed = False
    phase('3: Running pipeline')
    for name in names:
        result_file = run_dir / name / 'results.json'
        code = code_identity(name)
        request = {'pipeline': name, 'config': configs[name], 'code': code, 'force': args.force,
                   'pred_root': str(pred_root / name), 'results': str(result_file),
                   # Do not leak reference audio paths to inference workers.
                   'samples': [{'key': row['key'], 'mixture': row['mixture']} for row in prepared if not row.get('preparation_error')]}
        exit_code = launch_pipeline(name, request, run_dir)
        failed |= exit_code != 0
        try:
            results = {row['key']: row for row in json.loads(result_file.read_text())}
        except (OSError, ValueError, TypeError, KeyError):
            results = {}
        runtime[name], scored[name] = [], []
        phase(f'4: Running benchmark ({name})')
        for sample in prepared:
            key = sample['key']
            row = results.get(key, {'key': key, 'status': 'failed', 'error': sample.get('preparation_error') or f'Worker did not finish sample (exit={exit_code}); see worker.log'})
            try:
                if row['status'] == 'complete':
                    identity = fingerprint(name, Path(sample['mixture']), configs[name], code)
                    if reusable(pred_root / name / key, identity, Path(sample['mixture'])) is None:
                        row = {**row, 'status': 'failed', 'error': 'Prediction manifest or tracks failed validation'}
                if row['status'] == 'complete':
                    score = benchmark.score_reference_sample(sample, pred_root / name, args.sample_rate,
                                                              args.vad_threshold_db, args.crosstalk_threshold_db)
                else:
                    failed = True
                    score = benchmark._null_reference_row(key, 'pipeline_failed', {'pipeline': row.get('error', 'Unknown error')})
            except Exception as exc:
                score = benchmark._null_reference_row(key, 'benchmark_failed', {'benchmark': f'{type(exc).__name__}: {exc}'})
            runtime[name].append(row)
            failed |= score['status'] != 'ok'
            scored[name].append(score)
        report_path = report_base.parent / name / report_base.name
        benchmark.write_reference_reports(scored[name], report_path, title=f'OtoSpeech / {name} Reference Benchmark', print_table=True)
        write_json(report_path.parent / 'runtime.json', runtime[name])
        print(f'Saved benchmark: {report_path}', flush=True)
    comparison_report(scored, runtime, report_base.parent / 'comparison')
    write_json(run_dir / 'run.json', {'status': 'failed' if failed else 'complete', 'pipelines': list(names),
                                    'reports': str(report_base.parent), 'pred_root': str(pred_root)})
    return int(failed)
