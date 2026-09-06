"""ClearVoice bridge: NumPy 1.x models remain isolated from pyannote 4 / NumPy 2."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class IsolatedClearVoice:
    def __init__(self, model_name, device):
        import torch
        requested = str(device or 'auto')
        if requested in {'auto', 'cuda', 'gpu'}:
            self.resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif requested.startswith('cuda:'):
            self.resolved_device = requested if torch.cuda.is_available() else 'cpu'
        elif requested == 'cpu':
            self.resolved_device = 'cpu'
        else:
            raise ValueError(f'Unsupported ClearVoice device: {requested}')
        self.model_name = model_name
        self.child = None
        self.directory = tempfile.TemporaryDirectory(prefix='vilier-clearvoice-')

    def separate(self, audio, sample_rate):
        import numpy as np
        if self.child is None:
            project = ROOT / 'pipelines/vilier/clearvoice-runtime'
            env = dict(os.environ)
            env.pop('VIRTUAL_ENV', None)
            env.pop('PYTHONPATH', None)
            env['UV_PROJECT_ENVIRONMENT'] = str(project / '.venv')
            # ClearVoice chooses its own CUDA device; scope visible GPUs before import.
            env['CUDA_VISIBLE_DEVICES'] = '' if self.resolved_device == 'cpu' else self.resolved_device.partition(':')[2] or '0'
            self.child = subprocess.Popen(
                ['uv', 'run', '--project', str(project), '--locked', '--no-dev', 'python',
                 str(ROOT / 'comparison/clearvoice_worker.py'), '--model', self.model_name],
                cwd=project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1,
            )
        directory = Path(self.directory.name)
        np.save(directory / 'input.npy', audio, allow_pickle=False)
        self.child.stdin.write(json.dumps({'directory': str(directory), 'sample_rate': sample_rate}) + '\n')
        self.child.stdin.flush()
        while True:
            line = self.child.stdout.readline()
            if not line:
                raise RuntimeError(f'ClearVoice worker exited ({self.child.wait()})')
            if line.startswith('RESULT '):
                result = json.loads(line[7:])
                if result.get('error'):
                    raise RuntimeError(result['error'])
                return tuple(np.load(directory / f'{i}.npy', allow_pickle=False) for i in range(2))
            print(line, end='', flush=True)

    def close(self):
        if self.child is not None:
            try:
                self.child.stdin.close()
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            finally:
                self.child.stdout.close()
        self.directory.cleanup()
