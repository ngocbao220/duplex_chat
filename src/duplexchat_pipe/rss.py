from __future__ import annotations

import datetime as dt
from typing import Iterator
from urllib.parse import urlparse, urlunparse

import feedparser
import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wav",
    ".flac",
    ".m4b",
    ".webm",
    ".aif",
    ".aiff",
}


def fetch_rss(url: str, timeout_seconds: int) -> feedparser.FeedParserDict:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    return feedparser.parse(response.content)


def is_audio_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in AUDIO_EXTENSIONS)


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    normalized = parsed._replace(scheme=scheme, netloc=netloc, fragment="")
    return urlunparse(normalized)


def _iter_entry_audio_urls(entry: feedparser.FeedParserDict) -> Iterator[str]:
    enclosures = entry.get("enclosures", []) or []
    for enclosure in enclosures:
        href = enclosure.get("href")
        if href:
            yield href

    links = entry.get("links", []) or []
    for link in links:
        if link.get("rel") == "enclosure":
            href = link.get("href")
            if href:
                yield href

    media_content = entry.get("media_content", []) or []
    for media in media_content:
        href = media.get("url") or media.get("href")
        if href:
            yield href


def iter_audio_urls(feed: feedparser.FeedParserDict) -> Iterator[tuple[str, dict]]:
    for entry in feed.get("entries", []) or []:
        entry_meta = to_jsonable(entry)
        for audio_url in _iter_entry_audio_urls(entry):
            yield audio_url, entry_meta


def to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "items"):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)
