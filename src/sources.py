from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
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
    """Load source allowlist records and deduplicate by normalized URL."""
    if path is None or not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if path.suffix == ".json" or (path.suffix != ".jsonl" and stripped.startswith("[")):
        return _dedupe_records(_records_from_structured_json(path, text))
    return _dedupe_records(_records_from_jsonl(path, text))


def _records_from_jsonl(path: Path, text: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        raw = json.loads(line)
        records.append(_record_from_flat_mapping(path, raw, line_no))
    return records


def _records_from_structured_json(path: Path, text: str) -> list[SourceRecord]:
    raw = json.loads(text)
    sources = raw.get("sources", []) if isinstance(raw, dict) else raw
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        raise ValueError(f"{path}: expected a JSON object with a sources list")

    records: list[SourceRecord] = []
    for source_index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            raise ValueError(f"{path}: sources[{source_index}] must be an object")
        records.extend(_records_from_structured_source(path, source, source_index))
    return records


def _record_from_flat_mapping(path: Path, raw: dict[str, Any], index: int) -> SourceRecord:
    source_type = str(raw.get("source_type", "rss")).lower()
    if source_type not in {"rss", "youtube"}:
        raise ValueError(f"{path}:{index}: unsupported source_type {source_type!r}")
    return SourceRecord(
        source_id=str(raw.get("source_id") or f"allowlist_{index}"),
        source_type=source_type,
        url=normalize_url(str(raw["url"])),
        language=str(raw.get("language") or "vi").lower(),
        show_name=str(raw.get("show_name") or ""),
        license_notes=str(raw.get("license_notes") or ""),
        priority=int(raw.get("priority") or 0),
    )


def _records_from_structured_source(path: Path, source: dict[str, Any], index: int) -> list[SourceRecord]:
    base = {
        "source_id": str(source.get("source_id") or f"allowlist_{index}"),
        "language": str(source.get("language") or "vi").lower(),
        "show_name": str(source.get("show_name") or ""),
        "license_notes": str(source.get("license_notes") or ""),
        "priority": int(source.get("priority") or 0),
    }

    records: list[SourceRecord] = []
    for source_type, url in _iter_source_urls(source):
        records.append(
            SourceRecord(
                source_id=base["source_id"],
                source_type=source_type,
                url=normalize_url(url),
                language=base["language"],
                show_name=base["show_name"],
                license_notes=base["license_notes"],
                priority=base["priority"],
            )
        )
    return records


def _iter_source_urls(source: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for url in _as_url_list(source.get("url")):
        yield (str(source.get("source_type") or "rss").lower(), url)
    for url in _as_url_list(source.get("urls")):
        yield (str(source.get("source_type") or "rss").lower(), url)

    youtube = source.get("youtube")
    if isinstance(youtube, dict):
        for key in ("playlists", "channels", "videos", "urls"):
            for url in _as_url_list(youtube.get(key)):
                yield "youtube", url

    rss = source.get("rss")
    if isinstance(rss, dict):
        for url in _as_url_list(rss.get("urls")):
            yield "rss", url


def _as_url_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _dedupe_records(records: Iterable[SourceRecord]) -> list[SourceRecord]:
    deduped: list[SourceRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.source_type not in {"rss", "youtube"}:
            raise ValueError(f"unsupported source_type {record.source_type!r}")
        if record.url in seen:
            continue
        seen.add(record.url)
        deduped.append(record)
    return deduped
