from __future__ import annotations

from .models import ActivitySegment, Region


def _is_active_at(segments: list[ActivitySegment], start: float, end: float) -> bool:
    return any(segment.start < end and segment.end > start for segment in segments)


def _shortest_covering_duration(
    segments: list[ActivitySegment],
    start: float,
    end: float,
) -> float | None:
    durations = [
        segment.duration
        for segment in segments
        if segment.start < end and segment.end > start
    ]
    return min(durations) if durations else None


def classify_regions(
    mask_0: list[ActivitySegment],
    mask_1: list[ActivitySegment],
    duration_sec: float,
    backchannel_max_duration: float = 1.0,
) -> list[Region]:
    boundaries = {0.0, float(duration_sec)}
    for segment in [*mask_0, *mask_1]:
        boundaries.add(max(0.0, min(float(duration_sec), segment.start)))
        boundaries.add(max(0.0, min(float(duration_sec), segment.end)))
    ordered = sorted(boundaries)

    regions: list[Region] = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        active_0 = _is_active_at(mask_0, start, end)
        active_1 = _is_active_at(mask_1, start, end)
        if not active_0 and not active_1:
            region = Region(start, end, "silence")
        elif active_0 and not active_1:
            region = Region(start, end, "single_speaker", speaker=0)
        elif active_1 and not active_0:
            region = Region(start, end, "single_speaker", speaker=1)
        else:
            duration_0 = _shortest_covering_duration(mask_0, start, end) or 0.0
            duration_1 = _shortest_covering_duration(mask_1, start, end) or 0.0
            if min(duration_0, duration_1) <= backchannel_max_duration:
                backchannel_speaker = 0 if duration_0 <= duration_1 else 1
                region = Region(
                    start,
                    end,
                    "overlap_backchannel",
                    backchannel_speaker=backchannel_speaker,
                )
            else:
                region = Region(start, end, "overlap")
        if regions and _can_merge(regions[-1], region):
            previous = regions.pop()
            regions.append(
                Region(
                    previous.start,
                    region.end,
                    previous.type,
                    speaker=previous.speaker,
                    backchannel_speaker=previous.backchannel_speaker,
                )
            )
        else:
            regions.append(region)
    return regions


def _can_merge(left: Region, right: Region) -> bool:
    return (
        left.type == right.type
        and left.speaker == right.speaker
        and left.backchannel_speaker == right.backchannel_speaker
    )
