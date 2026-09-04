from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ActivitySegment:
    start: float
    end: float
    speaker: int
    label: str = "1"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    type: str
    speaker: int | None = None
    backchannel_speaker: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}
