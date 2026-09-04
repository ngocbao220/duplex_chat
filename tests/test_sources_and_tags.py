from pathlib import Path
import json

from duplexchat_pipe.config import Config
from duplexchat_pipe.pipeline import _collect_source_items
from duplexchat_pipe.sources import load_allowlist
from duplexchat_pipe.sources import SourceRecord
from duplexchat_pipe.tags import is_music_feed


def test_vietnamese_music_keywords_are_rejected():
    feed = {"tags": [{"term": "Nhạc và karaoke"}]}
    assert is_music_feed(feed)


def test_allowlist_parser_dedups_normalized_urls(tmp_path: Path):
    path = tmp_path / "allowlist.jsonl"
    path.write_text(
        '{"source_id":"a","source_type":"rss","url":"HTTPS://Example.com/feed.xml#x","language":"vi"}\n'
        '{"source_id":"b","source_type":"rss","url":"https://example.com/feed.xml","language":"vi-vn"}\n',
        encoding="utf-8",
    )

    records = load_allowlist(path)

    assert len(records) == 1
    assert records[0].source_id == "a"
    assert records[0].url == "https://example.com/feed.xml"


def test_allowlist_parser_loads_structured_youtube_sources(tmp_path: Path):
    path = tmp_path / "youtube_allowlist.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "show_youtube",
                        "show_name": "Show",
                        "language": "vi",
                        "priority": 10,
                        "youtube": {
                            "playlists": [
                                "HTTPS://Youtube.com/playlist?list=abc#section",
                                "https://youtube.com/playlist?list=abc",
                            ],
                            "channels": ["https://youtube.com/@show"],
                            "videos": ["https://youtube.com/watch?v=demo"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_allowlist(path)

    assert len(records) == 3
    assert {record.source_type for record in records} == {"youtube"}
    assert {record.source_id for record in records} == {"show_youtube"}
    assert {record.show_name for record in records} == {"Show"}
    assert records[0].url == "https://youtube.com/playlist?list=abc"


def test_collect_source_items_expands_youtube_records(monkeypatch):
    def fake_iter_entries(url: str, limit: int | None, target_duration_sec: float | None):
        assert url == "https://youtube.com/playlist?list=abc"
        assert limit == 2
        assert target_duration_sec == 3600
        return [
            {"id": "v1", "title": "Video 1", "url": "https://youtube.com/watch?v=v1"},
            {"id": "v2", "title": "Video 2", "url": "https://youtube.com/watch?v=v2"},
        ]

    monkeypatch.setattr("duplexchat_pipe.pipeline.youtube.iter_entries", fake_iter_entries)
    cfg = Config(episode_limit_per_feed=2, target_hours=1, youtube_only=True)
    record = SourceRecord(
        source_id="show_youtube",
        source_type="youtube",
        url="https://youtube.com/playlist?list=abc",
        language="vi",
        show_name="Show",
        license_notes="public source",
        priority=10,
    )

    items = _collect_source_items(record, cfg)

    assert len(items) == 2
    assert items[0].source_type == "youtube"
    assert items[0].audio_url == "https://youtube.com/watch?v=v1"
    assert items[0].feed_meta["source_id"] == "show_youtube"
    assert items[0].entry_meta["title"] == "Video 1"
