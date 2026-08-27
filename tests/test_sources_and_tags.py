from pathlib import Path

from duplexchat_pipe.sources import load_allowlist
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
