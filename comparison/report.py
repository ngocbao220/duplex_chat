from __future__ import annotations

import math
import statistics
from pathlib import Path

from .contract import write_json

# The names match the actual reference implementations, not proxy/model metrics.
METRICS = [
    ('all', 'pit_si_sdr', 'SI-SDR (dB)'), ('all', 'delta_si_sdr', 'SI-SDRi (dB)'),
    ('non_overlap', 'pit_si_sdr', 'Non-overlap SI-SDR (dB)'),
    ('overlap', 'pit_si_sdr', 'Overlap SI-SDR (dB)'),
    ('all', 'sar', 'SAR (dB)'), ('overlap', 'sir', 'Overlap SIR (dB)'),
    ('all', 'stoi', 'ESTOI'), ('all', 'pesq', 'PESQ'),
    ('overlap', 'crosstalk_rate', 'Crosstalk rate'), ('all', 'vad_f1', 'VAD F1'),
    ('all', 'onset_mae', 'Onset MAE (s)'), ('all', 'offset_mae', 'Offset MAE (s)'),
    ('overlap', 'overlap_f1', 'Overlap F1'), ('overlap', 'overlap_iou', 'Overlap IoU'),
]


def comparison_report(scored: dict[str, list[dict]], runtime: dict[str, list[dict]], output: Path):
    valid = [{row['key'] for row in rows if row['status'] == 'ok'} for rows in scored.values()]
    common = set.intersection(*valid) if valid else set()
    summaries = {}
    for name, rows in scored.items():
        selected = [row for row in rows if row['key'] in common and row['status'] == 'ok']
        stats = {}
        for condition, metric, label in METRICS:
            values = [row.get(condition, {}).get(metric) for row in selected]
            values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
            stats[label] = {'mean': statistics.mean(values) if values else None, 'available': len(values)}
        run_rows = runtime[name]
        completed = [row for row in run_rows if row['status'] == 'complete']
        # Work from this invocation only; historical runtimes must not look like new inference.
        fresh = [row for row in completed if not row.get('resumed')]
        seconds = sum(row.get('inference_seconds', 0) for row in fresh)
        duration = sum(row.get('duration_sec', 0) for row in fresh)
        summaries[name] = {'metrics': stats, 'successful': len(completed),
                           'failed': len(run_rows) - len(completed),
                           'resumed': len(completed) - len(fresh), 'inference_seconds': seconds,
                           'rtf': seconds / duration if duration else None}
    payload = {'dataset': 'OtoSpeech', 'common_keys': sorted(common), 'common_samples': len(common), 'pipelines': summaries}
    write_json(output.with_suffix('.json'), payload)
    names = list(summaries)
    lines = ['# OtoSpeech pipeline comparison', '', f'Common valid samples: {len(common)}', '',
             '| Metric | ' + ' | '.join(names) + ' |', '| --- | ' + ' | '.join('---' for _ in names) + ' |']
    for field in ('successful', 'failed', 'resumed', 'inference_seconds', 'rtf'):
        values = [format_value(summaries[name][field]) for name in names]
        lines.append('| ' + field + ' | ' + ' | '.join(values) + ' |')
    for _, _, label in METRICS:
        cells = [f"{format_value(summaries[name]['metrics'][label]['mean'])} (n={summaries[name]['metrics'][label]['available']})" for name in names]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    output.with_suffix('.md').write_text(text)
    print(text, flush=True)
    return payload


def format_value(value):
    if value is None:
        return 'N/A'
    return f'{value:.4f}' if isinstance(value, float) else str(value)
