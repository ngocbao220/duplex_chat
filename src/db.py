from __future__ import annotations

import logging
import sqlite3
import tarfile
from pathlib import Path
from typing import Iterator

import requests


LOGGER = logging.getLogger(__name__)


def download_feeds_db(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = cache_dir / "podcastindex_feeds.db.tgz"
    if tgz_path.exists() and tgz_path.stat().st_size > 0:
        return tgz_path

    LOGGER.info("Downloading feeds DB: %s", url)
    headers_list = [
        {"User-Agent": "duplexchat-pipe/0.1", "Accept": "*/*"},
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Referer": "https://public.podcastindex.org/",
        },
    ]

    session = requests.Session()
    last_exc: Exception | None = None
    for headers in headers_list:
        try:
            with session.get(
                url, stream=True, timeout=60, headers=headers, allow_redirects=True
            ) as response:
                if response.status_code == 403:
                    warmup = session.get(
                        "https://public.podcastindex.org/",
                        headers=headers,
                        timeout=30,
                    )
                    warmup.close()
                    with session.get(
                        url,
                        stream=True,
                        timeout=60,
                        headers=headers,
                        allow_redirects=True,
                    ) as retry:
                        if retry.status_code == 403:
                            continue
                        retry.raise_for_status()
                        with tgz_path.open("wb") as handle:
                            for chunk in retry.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                        return tgz_path

                response.raise_for_status()
                with tgz_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                return tgz_path
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to download feeds DB")


def _safe_extract(tar: tarfile.TarFile, member: tarfile.TarInfo, extract_dir: Path) -> Path:
    member_path = extract_dir / member.name
    resolved = member_path.resolve()
    if not str(resolved).startswith(str(extract_dir.resolve())):
        raise RuntimeError("Blocked unsafe tar path: %s" % member.name)
    tar.extract(member, path=extract_dir)
    return resolved


def extract_sqlite_db(tgz_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        db_members = [m for m in tar.getmembers() if m.name.endswith(".db")]
        if not db_members:
            raise RuntimeError("No .db file found in feeds archive")
        db_member = db_members[0]
        db_path = _safe_extract(tar, db_member, extract_dir)
    return db_path


def _choose_column(columns: list[str], preferred: list[str]) -> str | None:
    lookup = {col.lower(): col for col in columns}
    for pref in preferred:
        if pref in lookup:
            return lookup[pref]
    return None


def find_feed_table(conn: sqlite3.Connection) -> tuple[str, str, str]:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    best = None
    best_score = -1

    for table in tables:
        cols_cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cols_cursor.fetchall()]
        url_col = _choose_column(
            columns,
            ["feedurl", "feed_url", "feedUrl", "url", "originalurl", "original_url"],
        )
        lang_col = _choose_column(columns, ["language", "lang"])
        if not url_col or not lang_col:
            continue

        score = 5
        name_lower = table.lower()
        if "feed" in name_lower:
            score += 3
        if "podcast" in name_lower:
            score += 1
        if score > best_score:
            best = (table, url_col, lang_col)
            best_score = score

    if best is None:
        raise RuntimeError("Could not find a feed table with url+language columns")
    return best


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return ""
    return str(lang).strip().lower().replace("_", "-")


def lang_matches(lang: str, allowed: list[str]) -> bool:
    for code in allowed:
        code_norm = normalize_lang(code)
        if not code_norm:
            continue
        if lang == code_norm or lang.startswith(code_norm + "-"):
            return True
    return False


def iter_feed_urls(db_path: Path, languages: list[str]) -> Iterator[tuple[str, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        table, url_col, lang_col = find_feed_table(conn)
        query = f"SELECT {url_col}, {lang_col} FROM {table}"
        for url, lang in conn.execute(query):
            if not url:
                continue
            lang_norm = normalize_lang(lang)
            if not lang_norm:
                continue
            if lang_matches(lang_norm, languages):
                yield str(url).strip(), lang_norm
    finally:
        conn.close()
