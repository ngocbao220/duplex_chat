from duplexchat_pipe.dialogue import split_into_dialogues, is_balanced_dialogue, extract_valid_dialogues, Dialogue


def seg(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


# ── split_into_dialogues ──────────────────────────────────────────────────────

def test_split_empty():
    assert split_into_dialogues([], gap_seconds=5.0) == []


def test_split_single_segment():
    result = split_into_dialogues([seg("A", 0, 10)], gap_seconds=5.0)
    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].end == 10


def test_split_no_gap():
    """Segments with gaps < 5 s stay in the same dialogue."""
    segments = [seg("A", 0, 10), seg("B", 13, 20)]  # gap = 3 s
    result = split_into_dialogues(segments, gap_seconds=5.0)
    assert len(result) == 1


def test_split_exact_gap():
    """A gap of exactly 5 s triggers a split."""
    segments = [seg("A", 0, 10), seg("B", 15, 20)]  # gap = 5 s
    result = split_into_dialogues(segments, gap_seconds=5.0)
    assert len(result) == 2
    assert result[0].end == 10
    assert result[1].start == 15


def test_split_large_gap():
    segments = [seg("A", 0, 10), seg("B", 20, 30)]  # gap = 10 s
    result = split_into_dialogues(segments, gap_seconds=5.0)
    assert len(result) == 2


def test_split_multiple_dialogues():
    segments = [
        seg("A", 0, 5),
        seg("B", 6, 10),   # gap 1 s — same dialogue
        seg("A", 20, 25),  # gap 10 s — new dialogue
        seg("B", 26, 30),  # gap 1 s — same dialogue
    ]
    result = split_into_dialogues(segments, gap_seconds=5.0)
    assert len(result) == 2
    assert result[0].start == 0 and result[0].end == 10
    assert result[1].start == 20 and result[1].end == 30


# ── is_balanced_dialogue ──────────────────────────────────────────────────────

def test_balanced_equal_split():
    dlg = Dialogue(segments=[seg("A", 0, 5), seg("B", 5, 10)], start=0, end=10)
    assert is_balanced_dialogue(dlg, max_single_speaker_ratio=0.8)


def test_balanced_exactly_at_threshold():
    dlg = Dialogue(segments=[seg("A", 0, 8), seg("B", 8, 10)], start=0, end=10)
    assert is_balanced_dialogue(dlg, max_single_speaker_ratio=0.8)


def test_unbalanced_above_threshold():
    dlg = Dialogue(segments=[seg("A", 0, 9), seg("B", 9, 10)], start=0, end=10)
    assert not is_balanced_dialogue(dlg, max_single_speaker_ratio=0.8)


# ── extract_valid_dialogues ───────────────────────────────────────────────────

def test_extract_empty_input():
    assert extract_valid_dialogues([], gap_seconds=5.0) == []


def test_extract_two_speakers_all_valid():
    segments = [
        seg("A", 0, 5), seg("B", 5, 10),
        seg("A", 20, 25), seg("B", 25, 30),
    ]
    result = extract_valid_dialogues(segments, gap_seconds=5.0, max_single_speaker_ratio=0.8)
    assert len(result) == 2


def test_extract_splits_on_third_speaker():
    """
    A B A C A B → runs: [A,B,A] then C starts a new run with A → [C,A],
    then B becomes the 3rd speaker closing [C,A]. Final: 2 runs.
    """
    segments = [
        seg("A", 0, 2), seg("B", 2, 4), seg("A", 4, 6),  # run 1: A+B (0-6)
        seg("C", 6, 8), seg("A", 8, 10),                  # run 2: C+A (6-10)
        seg("B", 10, 12),                                  # B breaks C+A; B alone → no run
    ]
    result = extract_valid_dialogues(
        segments, gap_seconds=30.0, max_single_speaker_ratio=0.8, min_duration_seconds=1.0
    )
    assert len(result) == 2
    assert result[0].start == 0 and result[0].end == 6
    assert result[1].start == 6 and result[1].end == 10
    assert {s["speaker"] for s in result[1].segments} == {"C", "A"}


def test_extract_four_speaker_dialogue_yields_runs():
    """From a 4-speaker dialogue, extract the 2-speaker sub-runs."""
    segments = [
        seg("A", 0, 3), seg("B", 3, 6),    # run 1: A+B
        seg("C", 6, 8),                     # C breaks it; C alone → no run
        seg("D", 8, 10), seg("C", 10, 13), # run 2: D+C
        seg("A", 13, 15),                   # A breaks D+C
    ]
    result = extract_valid_dialogues(
        segments, gap_seconds=30.0, max_single_speaker_ratio=0.8, min_duration_seconds=1.0
    )
    assert len(result) == 2
    assert {s["speaker"] for s in result[0].segments} == {"A", "B"}
    assert {s["speaker"] for s in result[1].segments} == {"D", "C"}


def test_extract_dominance_filter_applied_to_runs():
    """A run with 2 speakers but 90%/10% split is rejected."""
    segments = [
        seg("A", 0, 9), seg("B", 9, 10),   # A=90% → rejected
        seg("A", 20, 25), seg("B", 25, 30), # A=50% → kept
    ]
    result = extract_valid_dialogues(segments, gap_seconds=5.0, max_single_speaker_ratio=0.8)
    assert len(result) == 1
    assert result[0].start == 20
