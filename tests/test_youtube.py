import json
import subprocess

from duplexchat_pipe import youtube


def test_iter_entries_stops_after_target_duration(monkeypatch):
    lines = [
        {"id": "a", "url": "a", "duration": 1200, "title": "A"},
        {"id": "b", "url": "b", "duration": 1500, "title": "B"},
        {"id": "c", "url": "c", "duration": 1200, "title": "C"},
        {"id": "d", "url": "d", "duration": 900, "title": "D"},
    ]

    monkeypatch.setattr(youtube, "ensure_ytdlp", lambda: None)
    monkeypatch.setattr(youtube, "_ytdlp_cmd", lambda: ["yt-dlp"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="\n".join(json.dumps(line) for line in lines),
            stderr="",
        ),
    )

    entries = youtube.iter_entries("https://youtube.com/playlist?list=abc", target_duration_sec=3600)

    assert [entry["id"] for entry in entries] == ["a", "b", "c"]
    assert entries[-1]["url"] == "https://www.youtube.com/watch?v=c"
