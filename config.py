from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AspectRatio(Enum):
    PORTRAIT = "9:16"   # 1080x1920
    LANDSCAPE = "16:9"  # 1920x1080


class Theme(Enum):
    NEON = "neon"
    VINYL = "vinyl"
    WAVE = "wave"


class LyricsStyle(Enum):
    KARAOKE = "karaoke"
    FADE = "fade"
    WORD_FILL = "word-fill"


FPS = 30

RESOLUTIONS = {
    AspectRatio.PORTRAIT: (1080, 1920),
    AspectRatio.LANDSCAPE: (1920, 1080),
}

# Timeline constants (seconds)
COVER_DURATION = 3.0
TRANSITION_DURATION = 2.0
OUTRO_DURATION = 3.0


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


@dataclass
class TimedLine:
    text: str
    start: float
    end: float
    words: list[TimedWord] = field(default_factory=list)


@dataclass
class AudioFeatures:
    """Per-frame audio features, arrays of length num_frames."""
    rms: list[float]           # RMS volume per frame, normalized 0-1
    spectrum: list[list[float]]  # Frequency bands per frame
    beat_frames: list[int]     # Frame indices where beats occur
    duration: float            # Total duration in seconds


class GenerateMode(Enum):
    FULL = "full"       # Full song
    CHORUS = "chorus"   # Auto-detect chorus section


@dataclass
class GenerateConfig:
    audio_path: Path
    lyrics_path: Path
    cover_path: Path
    output_path: Path = Path("output.mp4")
    aspect: AspectRatio = AspectRatio.PORTRAIT
    theme: Theme = Theme.NEON
    lyrics_style: LyricsStyle = LyricsStyle.KARAOKE
    mode: GenerateMode = GenerateMode.FULL
    title: str = ""
    artist: str = ""
