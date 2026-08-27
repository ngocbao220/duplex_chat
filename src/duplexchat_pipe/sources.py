from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from duplexchat_pipe.rss import normalize_url


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_type: str
    url: str
    language: str
    show_name: str = ""
    license_notes: str = ""
    priority: int = 0


def load_allowlist(path: Path | None) -> list[SourceRecord]:
    """Load JSONL allowlist records and deduplicate by normalized URL."""
    if path is None or not path.exists():
        return []

    records: list[SourceRecord] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        raw = json.loads(line)
        url = normalize_url(str(raw["url"]))
        if url in seen:
            continue
        seen.add(url)
        source_type = str(raw.get("source_type", "rss")).lower()
        if source_type not in {"rss", "youtube"}:
            raise ValueError(f"{path}:{line_no}: unsupported source_type {source_type!r}")
        records.append(
            SourceRecord(
                source_id=str(raw.get("source_id") or f"allowlist_{line_no}"),
                source_type=source_type,
                url=url,
                language=str(raw.get("language") or "vi").lower(),
                show_name=str(raw.get("show_name") or ""),
                license_notes=str(raw.get("license_notes") or ""),
                priority=int(raw.get("priority") or 0),
            )
        )
    return records
