from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def make_run_id() -> str:
    return dt.datetime.now().strftime("%H%M%S")


def setup_run_logging(log_root: Path, run_id: str | None = None) -> Path:
    date = dt.date.today().isoformat()
    effective_run_id = run_id or make_run_id()
    run_dir = Path(log_root) / date / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(phase)s] %(message)s"
    )
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    root.addHandler(_PhaseDefaultHandler(file_handler))
    root.addHandler(_PhaseDefaultHandler(console_handler))

    for name, content in {
        "phase_tree.txt": "",
        "stats_table.md": "",
        "errors.jsonl": "",
        "artifacts.json": "{}\n",
    }.items():
        path = run_dir / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return run_dir


class _PhaseDefaultHandler(logging.Handler):
    def __init__(self, wrapped: logging.Handler) -> None:
        super().__init__(wrapped.level)
        self.wrapped = wrapped

    def emit(self, record: logging.LogRecord) -> None:
        if not hasattr(record, "phase"):
            record.phase = "-"
        self.wrapped.emit(record)


def dataclass_to_dict(value: Any) -> dict:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def write_resolved_config(run_dir: Path, cfg: Any) -> None:
    data = dataclass_to_dict(cfg)
    lines = []
    for key in sorted(data):
        val = data[key]
        if isinstance(val, Path):
            val = str(val)
        lines.append(f"{key}: {json.dumps(val, ensure_ascii=False, default=str)}")
    (run_dir / "resolved_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_phase_tree(run_dir: Path, phase: str, rows: list[tuple[str, str]]) -> None:
    lines = [f"{phase}"]
    for idx, (label, value) in enumerate(rows):
        branch = "`--" if idx == len(rows) - 1 else "|--"
        lines.append(f"{branch} {label}: {value}")
    with (run_dir / "phase_tree.txt").open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def append_stats_table(run_dir: Path, title: str, stats: dict[str, Any]) -> None:
    with (run_dir / "stats_table.md").open("a", encoding="utf-8") as fp:
        fp.write(f"\n## {title}\n\n| Metric | Value |\n|---|---:|\n")
        for key, value in stats.items():
            fp.write(f"| {key} | {value} |\n")


def append_error(run_dir: Path, payload: dict[str, Any]) -> None:
    with (run_dir / "errors.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_artifacts(run_dir: Path, artifacts: dict[str, Any]) -> None:
    (run_dir / "artifacts.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
