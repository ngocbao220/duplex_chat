from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='alibabasglab/MossFormer2_SS_16K')
    parser.add_argument('--check-imports', action='store_true')
    args = parser.parse_args()
    sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipelines/vilier'))
    if args.check_imports:
        from clearvoice import ClearVoice
        print('ClearVoice imports OK')
        return
    # Keep library prints off the response channel. stderr is inherited by the
    # outer worker, which already streams and persists it in worker.log.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from pipeline.overlap_separation import ClearVoiceSeparator
        model = ClearVoiceSeparator(args.model, 'auto')
    for line in sys.stdin:
        try:
            request = json.loads(line)
            directory = Path(request['directory'])
            with contextlib.redirect_stdout(sys.stderr):
                tracks = model.separate(np.load(directory / 'input.npy', allow_pickle=False), request['sample_rate'])
            for index, track in enumerate(tracks):
                np.save(directory / f'{index}.npy', track, allow_pickle=False)
            response = {'status': 'complete'}
        except Exception as exc:
            response = {'error': f'{type(exc).__name__}: {exc}'}
        print('RESULT ' + json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
