from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def stream_process(command: list[str], cwd: Path, log: Path, env: dict | None = None) -> int:
    """Stream merged stdout/stderr and reap the complete process group on interruption."""
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ if env is None else env)
    child_env.setdefault('MPLBACKEND', 'Agg')
    child_env['PYTHONUNBUFFERED'] = '1'
    child_env.setdefault('MPLCONFIGDIR', str(log.parent / 'matplotlib'))
    child_env.setdefault('NUMBA_CACHE_DIR', str(log.parent / 'numba'))
    with log.open('a', encoding='utf-8') as handle:
        with subprocess.Popen(command, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace',
                              start_new_session=True, bufsize=1) as child:
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    handle.write(line)
                    handle.flush()
                return child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    child.wait()
                raise
