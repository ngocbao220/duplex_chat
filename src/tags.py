"""Shared tag-based filtering helpers.

Used by the live crawl (skip music feeds before download) and by
post-processing (drop music samples during shard filtering).
"""

from __future__ import annotations

# Lowercase substring matches (case-insensitive). A feed whose tag term
# contains any of these strings is treated as music.
MUSIC_KEYWORDS: tuple[str, ...] = (
    "music", "songs", "song", "jazz", "rock", "classical",
    "pop", "hip-hop", "hip hop", "rap", "electronic", "edm",
    "reggae", "blues", "country", "metal", "folk", "r&b", "rnb",
    "soul", "punk", "indie", "alternative", "opera",
    "音楽", "楽曲", "ソング", "歌",
    "nhạc", "bài hát", "ca nhạc", "acoustic", "karaoke",
    "remix", "live session", "bolero", "vpop",
)


def _tag_terms(section: dict | None) -> list[str]:
    if not section:
        return []
    tags = section.get("tags") or []
    terms: list[str] = []
    for tag in tags:
        if isinstance(tag, dict):
            term = tag.get("term")
            if term:
                terms.append(str(term))
    return terms


def is_music_feed(feed_meta: dict | None, entry_meta: dict | None = None) -> bool:
    """Return True if any feed or entry tag term matches a music keyword."""
    for section in (feed_meta, entry_meta):
        for term in _tag_terms(section):
            lower = term.lower()
            if any(kw in lower for kw in MUSIC_KEYWORDS):
                return True
    return False
