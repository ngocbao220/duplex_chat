import sqlite3

from duplexchat_pipe.db import find_feed_table


def test_find_feed_table():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE feeds (id INTEGER PRIMARY KEY, feedurl TEXT, language TEXT)"
    )
    table, url_col, lang_col = find_feed_table(conn)
    assert table == "feeds"
    assert url_col == "feedurl"
    assert lang_col == "language"
    conn.close()
