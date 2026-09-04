from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Dialogue:
    segments: list[dict]
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speakers(self) -> list[str]:
        return list({s["speaker"] for s in self.segments})


def split_into_dialogues(segments: list[dict], gap_seconds: float) -> list[Dialogue]:
    """Group consecutive diarization segments; split when the gap between turns >= gap_seconds."""
    if not segments:
        return []

    groups: list[list[dict]] = []
    current: list[dict] = [segments[0]]

    for seg in segments[1:]:
        if seg["start"] - current[-1]["end"] >= gap_seconds:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)

    return [Dialogue(segments=g, start=g[0]["start"], end=g[-1]["end"]) for g in groups]


def _two_speaker_runs(segments: list[dict]) -> list[list[dict]]:
    """
    Find all maximal contiguous sub-sequences of turns that involve exactly
    2 distinct speakers. When a 3rd speaker appears, the current run is closed
    and a new one begins with just that speaker.
    """
    if not segments:
        return []

    runs: list[list[dict]] = []
    current: list[dict] = [segments[0]]
    speakers: set[str] = {segments[0]["speaker"]}

    for seg in segments[1:]:
        if seg["speaker"] in speakers or len(speakers) < 2:
            current.append(seg)
            speakers.add(seg["speaker"])
        else:
            if len(speakers) == 2:
                runs.append(current)
            current = [seg]
            speakers = {seg["speaker"]}

    if len(speakers) == 2:
        runs.append(current)

    return runs


def is_balanced_dialogue(dialogue: Dialogue, max_single_speaker_ratio: float) -> bool:
    """Return True if no single speaker exceeds max_single_speaker_ratio of total turn time."""
    speaker_duration: dict[str, float] = {}
    for seg in dialogue.segments:
        dur = seg["end"] - seg["start"]
        speaker_duration[seg["speaker"]] = speaker_duration.get(seg["speaker"], 0.0) + dur
    total = sum(speaker_duration.values())
    if total <= 0:
        return False
    return max(speaker_duration.values()) / total <= max_single_speaker_ratio


def _split_long_dialogue(
    dlg: Dialogue, max_duration: float, min_duration: float,
) -> list[Dialogue]:
    """Split a dialogue into chunks of at most max_duration seconds."""
    if dlg.duration <= max_duration:
        return [dlg]

    chunks: list[Dialogue] = []
    current: list[dict] = []
    chunk_start = dlg.segments[0]["start"]

    for seg in dlg.segments:
        current.append(seg)
        if seg["end"] - chunk_start >= max_duration:
            chunks.append(Dialogue(segments=current, start=chunk_start, end=seg["end"]))
            current = []
            chunk_start = seg["end"]

    if current:
        candidate = Dialogue(segments=current, start=chunk_start, end=current[-1]["end"])
        if candidate.duration >= min_duration:
            chunks.append(candidate)

    return chunks


def extract_valid_dialogues(
    segments: list[dict],
    gap_seconds: float = 5.0,
    max_single_speaker_ratio: float = 0.8,
    min_duration_seconds: float = 10.0,
    max_duration_seconds: float = 600.0,
) -> list[Dialogue]:
    """
    Split by silence gaps, then within each group extract all maximal 2-speaker
    runs and keep those that pass the dominance and duration filters.
    Dialogues longer than max_duration_seconds are split into chunks.
    """
    result: list[Dialogue] = []
    for group in split_into_dialogues(segments, gap_seconds):
        for run in _two_speaker_runs(group.segments):
            dlg = Dialogue(segments=run, start=run[0]["start"], end=run[-1]["end"])
            if dlg.duration < min_duration_seconds:
                continue
            for chunk in _split_long_dialogue(dlg, max_duration_seconds, min_duration_seconds):
                if is_balanced_dialogue(chunk, max_single_speaker_ratio):
                    result.append(chunk)
    return result
