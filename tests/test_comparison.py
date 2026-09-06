import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from comparison.contract import fingerprint, run_sample, validate_tracks, vilier_tracks
from comparison.process import stream_process


def wav(path, length=1600, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(length), rate)
    return path


def test_vilier_requires_exactly_two_full_timeline_tracks(tmp_path):
    source = wav(tmp_path / 'mixture.wav')
    tracks = [wav(tmp_path / f'{i}.wav') for i in range(2)]
    manifest = tmp_path / 'manifest.timeline.json'
    manifest.write_text(json.dumps({'speakers': [{'track_wav': p.name} for p in tracks]}))
    assert vilier_tracks(manifest) == tracks
    validate_tracks(source, tracks)
    wav(tracks[1], length=800)
    with pytest.raises(ValueError, match='timeline'):
        validate_tracks(source, tracks)
    manifest.write_text(json.dumps({'speakers': [{'track_wav': tracks[0].name}]}))
    with pytest.raises(ValueError, match='two'):
        vilier_tracks(manifest)


def test_resume_checks_input_config_and_outputs_and_preserves_history(tmp_path):
    source = wav(tmp_path / 'mixture.wav')
    output = tmp_path / 'prediction'
    calls = []
    def adapter(source, output, config):
        calls.append(config)
        return [wav(output / f'{i}.wav') for i in range(2)], {'device': 'cpu'}
    sample = {'key': 'a', 'mixture': str(source)}
    config = {'model': 'one'}
    first = run_sample('cholimex', sample, output, config, 'code1', adapter)
    assert first['status'] == 'complete'
    assert run_sample('cholimex', sample, output, config, 'code1', adapter)['resumed']
    assert len(calls) == 1
    run_sample('cholimex', sample, output, {'model': 'two'}, 'code1', adapter)
    assert len(calls) == 2
    run_sample('cholimex', sample, output, {'model': 'two'}, 'code1', adapter, force=True)
    assert len(calls) == 3
    assert list((tmp_path / '.history').rglob('run.json'))
    (output / 'speaker_A.wav').write_bytes(b'corrupt')
    run_sample('cholimex', sample, output, {'model': 'two'}, 'code1', adapter)
    assert len(calls) == 4
    before = fingerprint('cholimex', source, config, 'code1')
    wav(source, length=800)
    assert fingerprint('cholimex', source, config, 'code1') != before


def test_failed_adapter_does_not_leave_success(tmp_path):
    source = wav(tmp_path / 'mixture.wav')
    def fail(*args):
        raise RuntimeError('model unavailable')
    result = run_sample('vilier', {'key': 'a', 'mixture': str(source)}, tmp_path / 'out', {}, 'code1', fail)
    assert result['status'] == 'failed'
    assert 'model unavailable' in result['error']
    assert json.loads((tmp_path / 'out/run.json').read_text())['status'] == 'failed'


def test_subprocess_streams_and_returns_failure(tmp_path, capsys):
    log = tmp_path / 'worker.log'
    result = stream_process([sys.executable, '-c', 'print("worker output", flush=True); raise SystemExit(3)'], tmp_path, log)
    assert result == 3
    assert 'worker output' in capsys.readouterr().out
    assert 'worker output' in log.read_text()


def test_comparison_uses_only_common_samples_and_prints_na(tmp_path, capsys):
    from comparison.report import comparison_report
    scored = {'cholimex': [{'key': 'a', 'status': 'ok', 'all': {'pit_si_sdr': 1}},
                           {'key': 'b', 'status': 'ok', 'all': {'pit_si_sdr': 100}}],
              'vilier': [{'key': 'a', 'status': 'ok', 'all': {'pit_si_sdr': 3}},
                         {'key': 'b', 'status': 'pipeline_failed'}]}
    runtime = {name: [{'key': 'a', 'status': 'complete', 'inference_seconds': 2, 'duration_sec': 4}] for name in scored}
    result = comparison_report(scored, runtime, tmp_path / 'comparison')
    assert result['common_keys'] == ['a']
    assert result['pipelines']['cholimex']['metrics']['SI-SDR (dB)']['mean'] == 1
    assert result['pipelines']['cholimex']['rtf'] == .5
    assert 'N/A (n=0)' in capsys.readouterr().out
    assert json.loads((tmp_path / 'comparison.json').read_text()) == result
    scored['vilier'][0]['status'] = 'pipeline_failed'
    assert comparison_report(scored, runtime, tmp_path / 'empty')['common_samples'] == 0


@pytest.mark.parametrize('fail_pipeline', [None, 'duplexchat'])
def test_all_prepares_once_and_isolates_predictions(monkeypatch, tmp_path, fail_pipeline, capsys):
    import argparse
    import shutil
    from comparison import runner, worker
    from duplexchat_pipe import benchmark
    from duplexchat_pipe.config import Config
    from duplexchat_pipe.outputs import save_wav
    source = tmp_path / 'dataset'
    first, second = wav(source / 's1.wav'), wav(source / 's2.wav')
    downloads, mixes, workers = [], [], []
    monkeypatch.setattr(benchmark, 'download_otospeech_dataset', lambda **kwargs: downloads.append(kwargs) or source)
    monkeypatch.setattr(benchmark, 'discover_otospeech_samples', lambda root: [{'key': 'train/a', 'gt_speaker_1': str(first), 'gt_speaker_2': str(second)}])
    original_mix = benchmark.mix_ground_truth_pair
    def mix(*args):
        mixes.append(args)
        return original_mix(*args)
    monkeypatch.setattr(benchmark, 'mix_ground_truth_pair', mix)
    def launch(name, request, run_dir):
        workers.append(request)
        assert all(set(sample) == {'key', 'mixture'} for sample in request['samples'])
        def adapter(source, output, config):
            if name == fail_pipeline:
                raise RuntimeError('test model failure')
            tracks = [output / f'{i}.wav' for i in range(2)]
            for track in tracks:
                shutil.copy2(source, track)
            return tracks, {'device': 'cpu'}
        worker.run_batch(request, adapter)
        return int(name == fail_pipeline)
    monkeypatch.setattr(runner, 'launch_pipeline', launch)
    # The reference metric implementation has its own tests. This boundary keeps
    # the integration test focused on orchestration and counts mixture creation.
    monkeypatch.setattr(benchmark, 'score_reference_sample', lambda sample, *args: {'key': sample['key'], 'status': 'ok', 'all': {'pit_si_sdr': 2}})
    args = argparse.Namespace(pipeline='all', otospeech_root=None, otospeech_repo='test',
        otospeech_local_dir=None, size_gb=1, max_samples=None, max_seconds=None,
        sample_rate=16000, output_root=tmp_path / 'outputs', mixture_root=None,
        pred_root=None, benchmark_output=tmp_path / 'reports/summary.json',
        vilier_config=Path('configs/vilier.json'), duplexchat_config=Path('configs/duplexchat.json'),
        force=False, vad_threshold_db=-40, crosstalk_threshold_db=-20)
    assert runner.run_otospeech(Config(), args, benchmark, save_wav) == int(fail_pipeline is not None)
    assert len(downloads) == len(mixes) == 1
    assert [request['pipeline'] for request in workers] == list(runner.PIPELINES)
    assert len({request['samples'][0]['mixture'] for request in workers}) == 1
    copied = tmp_path / 'outputs/otospeech_mixtures/train/a'
    assert (copied / 'gt_speaker_1.wav').read_bytes() == first.read_bytes()
    assert (copied / 'gt_speaker_2.wav').read_bytes() == second.read_bytes()
    for name in runner.PIPELINES:
        result = json.loads((tmp_path / f'outputs/otospeech/{name}/train/a/run.json').read_text())
        assert result['status'] == ('failed' if name == fail_pipeline else 'complete')
        assert (tmp_path / f'reports/{name}/summary.md').is_file()
    assert 'OtoSpeech pipeline comparison' in capsys.readouterr().out


def test_process_reaps_group_on_interrupt(monkeypatch, tmp_path):
    import signal
    from comparison import process
    calls = []
    class BrokenOutput:
        def __iter__(self):
            raise KeyboardInterrupt()
    class Child:
        pid = 123
        stdout = BrokenOutput()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self, timeout=None): calls.append(('wait', timeout)); return 0
    monkeypatch.setattr(process.subprocess, 'Popen', lambda *args, **kwargs: Child())
    monkeypatch.setattr(process.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        process.stream_process(['unused'], tmp_path, tmp_path / 'interrupt.log')
    assert calls == [(123, signal.SIGTERM), ('wait', 10)]


def test_launch_scopes_uv_environment(monkeypatch, tmp_path):
    from comparison import runner
    seen = {}
    monkeypatch.setenv('UV_PROJECT_ENVIRONMENT', '/other/venv')
    monkeypatch.setenv('VIRTUAL_ENV', '/other/venv')
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    def launch(command, cwd, log, env):
        seen.update(command=command, cwd=cwd, env=env)
        return 0
    monkeypatch.setattr(runner, 'stream_process', launch)
    assert runner.launch_pipeline('vilier', {'pipeline': 'vilier'}, tmp_path) == 0
    assert seen['env']['UV_PROJECT_ENVIRONMENT'] == str(runner.ROOT / 'pipelines/vilier/.venv')
    assert 'VIRTUAL_ENV' not in seen['env']
    assert seen['env']['MPLBACKEND'] == 'Agg'
    assert '--locked' in seen['command']


def test_batch_continues_after_failed_sample(tmp_path):
    from comparison.worker import run_batch
    source = wav(tmp_path / 'mixture.wav')
    def adapter(source, output, config):
        if output.name == 'bad':
            raise RuntimeError('broken')
        return [wav(output / f'{i}.wav') for i in range(2)], {}
    request = {'pipeline': 'duplexchat', 'samples': [{'key': key, 'mixture': str(source)} for key in ['bad', 'good']],
               'pred_root': str(tmp_path / 'out'), 'config': {}, 'code': 'test', 'force': False,
               'results': str(tmp_path / 'results.json')}
    results = run_batch(request, adapter)
    assert [row['status'] for row in results] == ['failed', 'complete']


def test_single_audio_import_does_not_patch_torch_load():
    import importlib
    import torch
    from duplexchat_pipe import single_audio
    before = torch.load
    importlib.reload(single_audio)
    assert torch.load is before


def test_worker_script_does_not_shadow_legacy_runner_module():
    import subprocess
    result = subprocess.run([sys.executable, 'comparison/worker.py', '--check-imports', 'cholimex'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'cholimex imports OK' in result.stdout


def test_cholimex_worker_roundtrips_flat_config(monkeypatch, tmp_path):
    from dataclasses import asdict
    from comparison.worker import cholimex
    from duplexchat_pipe.config import Config
    import duplexchat_pipe.cholimex as module
    seen = []
    def run(source, output, cfg):
        seen.append(cfg)
        return {}
    monkeypatch.setattr(module, 'run_cholimex_file', run)
    config = json.loads(json.dumps(asdict(Config(runtime_device='cpu')), default=str))
    cholimex(tmp_path / 'in.wav', tmp_path / 'out', config)
    assert seen[0].runtime_device == 'cpu'
    assert isinstance(seen[0].output_dir, Path)


def test_clearvoice_bridge_transfers_audio_and_reaps_worker(monkeypatch):
    import subprocess
    from comparison.clearvoice import IsolatedClearVoice
    real_popen = subprocess.Popen
    children = []
    script = '''import json, sys, numpy as np
from pathlib import Path
for line in sys.stdin:
 r=json.loads(line); p=Path(r['directory']); x=np.load(p/'input.npy')
 np.save(p/'0.npy',x); np.save(p/'1.npy',-x)
 print('RESULT '+json.dumps({'status':'complete'}), flush=True)
'''
    def start(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        child = real_popen([sys.executable, '-u', '-c', script], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', start)
    helper = IsolatedClearVoice('test', 'cpu')
    try:
        audio = np.linspace(-.5, .5, 1600).astype('float32')
        first, second = helper.separate(audio, 16000)
        np.testing.assert_array_equal(first, audio)
        np.testing.assert_array_equal(second, -audio)
        helper.separate(audio, 16000)
        assert len(children) == 1
    finally:
        helper.close()
    assert children[0].poll() == 0
