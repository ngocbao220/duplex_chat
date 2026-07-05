from duplexchat_pipe.rss import is_audio_url


def test_is_audio_url_with_query():
    assert is_audio_url("https://example.com/audio.mp3")
    assert is_audio_url("https://example.com/audio.m4a?token=abc")
    assert is_audio_url("https://example.com/audio.MP3")
    assert not is_audio_url("https://example.com/page.html")
