import tarfile
from pathlib import Path

from duplexchat_pipe.wds import open_shard_writer, write_sample


def test_wds_schema(tmp_path: Path):
    writer = open_shard_writer(tmp_path, shard_size_gb=1)
    key = "abc123"
    write_sample(
        writer,
        key,
        b"mp3bytes",
        {"rss_url": "https://example.com"},
        {"segments": []},
        None,
    )
    writer.close()

    tar_path = tmp_path / "000000.tar.gz"
    assert tar_path.exists()
    with tarfile.open(tar_path, "r:gz") as tar:
        names = {m.name for m in tar.getmembers()}

    assert f"{key}.audio.mp3" in names
    assert f"{key}.meta.json" in names
    assert f"{key}.diarization.json" in names
