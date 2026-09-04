from duplexchat_pipe.logging_utils import setup_run_logging


def test_setup_run_logging_creates_video_stats_file(tmp_path):
    run_dir = setup_run_logging(tmp_path, "run")

    assert (run_dir / "video_stats.jsonl").exists()
