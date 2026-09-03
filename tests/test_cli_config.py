from pathlib import Path

from duplexchat_pipe.cli import build_parser


def test_run_parser_accepts_config_phase_and_overrides():
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "config.json",
            "--phase",
            "benchmark",
            "source.target_hours=2.5",
            "separation=mossformer2",
        ]
    )

    assert args.command == "run"
    assert args.config == Path("config.json")
    assert args.phase == "benchmark"
    assert args.overrides == ["source.target_hours=2.5", "separation=mossformer2"]
