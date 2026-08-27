from duplexchat_pipe.db import lang_matches, normalize_lang


def test_normalize_lang():
    assert normalize_lang("EN") == "en"
    assert normalize_lang("ja_JP") == "ja-jp"
    assert normalize_lang(None) == ""


def test_lang_matches():
    allowed = ["en", "ja"]
    assert lang_matches("en", allowed)
    assert lang_matches("en-us", allowed)
    assert lang_matches("ja", allowed)
    assert lang_matches("ja-jp", allowed)
    assert not lang_matches("fr", allowed)


def test_vietnamese_region_matches_base_language():
    assert lang_matches("vi-vn", ["vi"])
