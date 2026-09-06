"""One isolated process per pipeline, receiving mixtures only (never reference audio)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
from comparison.contract import run_sample, vilier_tracks, write_json


def cholimex(source, output, config):
    from duplexchat_pipe.config import Config, PATH_FIELDS
    from duplexchat_pipe.cholimex import run_cholimex_file
    from duplexchat_pipe.devices import resolve_device
    cfg = Config(**{key: Path(value) if key in PATH_FIELDS and value is not None else value
                    for key, value in config.items()})
    result = run_cholimex_file(source, output, cfg)
    return [output / 'speaker_0.wav', output / 'speaker_1.wav'], {
        'device': resolve_device(cfg.runtime_device, cfg.allow_cpu_fallback), 'pipeline_result': result}


def duplexchat(source, output, config):
    from duplexchat_pipe.single_audio import run_single_audio
    from duplexchat_pipe.devices import resolve_device
    run_single_audio(str(source), output_prefix=str(output / 'speaker'),
                     output_dir=str(output / 'phases'), **config)
    return [output / 'speaker_A.wav', output / 'speaker_B.wav'], {
        'device': resolve_device(config.get('runtime_device', 'auto'))}


def vilier(source, output, config):
    sys.path.insert(0, str(ROOT / 'pipelines/vilier'))
    from pipeline import cli
    from comparison.clearvoice import IsolatedClearVoice
    original_loader = cli.load_overlap_separator
    helpers = []

    def strict_loader(options, dry_run=False, warnings=None):
        if options.get('enabled') and options.get('backend') in {'clearvoice', 'mossformer2'}:
            helper = IsolatedClearVoice(options.get('model_name', 'alibabasglab/MossFormer2_SS_16K'), options.get('device', 'auto'))
            helpers.append(helper)
            return helper
        separator = original_loader(options, dry_run=dry_run, warnings=warnings)
        if options.get('enabled') and separator is None:
            raise RuntimeError('Vilier separation failed to load: ' + '; '.join(warnings or []))
        return separator

    cli.load_overlap_separator = strict_loader
    try:
        manifest, _ = cli.process_one(source, config, output / 'native', output / 'state',
                                      dry_run=False, until='pre_asr')
        native = json.loads(manifest.read_text())
        config_path = manifest.parent / 'run_config.json'
        devices = json.loads(config_path.read_text()) if config_path.exists() else native.get('run_config')
        return vilier_tracks(manifest), {'native_manifest': str(manifest), 'run_config': devices}
    finally:
        cli.load_overlap_separator = original_loader
        for helper in helpers:
            helper.close()



ADAPTERS = {'cholimex': cholimex, 'duplexchat': duplexchat, 'vilier': vilier}


def run_batch(request: dict, adapter=None) -> list[dict]:
    name = request['pipeline']
    adapter = adapter or ADAPTERS[name]
    results = []
    for index, sample in enumerate(request['samples'], 1):
        print(f"[{name} {index}/{len(request['samples'])}] Processing sample {sample['key']}", flush=True)
        result = run_sample(name, sample, Path(request['pred_root']) / sample['key'],
                            request['config'], request['code'], adapter, force=request['force'])
        results.append(result)
        write_json(Path(request['results']), results)
        print(f"[{name}] Complete sample {sample['key']}: {result['status']}"
              f"{' (resumed)' if result.get('resumed') else ''} {result.get('error', '')}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request', type=Path)
    group.add_argument('--check-imports', choices=list(ADAPTERS))
    args = parser.parse_args()
    if args.check_imports:
        if args.check_imports == 'vilier':
            sys.path.insert(0, str(ROOT / 'pipelines/vilier'))
            from pipeline import cli
        else:
            from duplexchat_pipe import single_audio, cholimex
        print(f'{args.check_imports} imports OK', flush=True)
        return 0
    request = json.loads(args.request.read_text())
    print(f"Pipeline={request['pipeline']} interpreter={sys.executable}", flush=True)
    print(json.dumps(request['config'], ensure_ascii=False, indent=2), flush=True)
    rows = run_batch(request)
    return int(any(row['status'] != 'complete' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
