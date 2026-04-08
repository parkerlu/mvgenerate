# MV Generate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI + Web tool that generates lyric music videos from MP3 + lyrics + cover image, with 3 visual themes and 3 lyrics display styles.

**Architecture:** Python core modules (lyrics preprocessing, Whisper alignment, audio analysis, frame rendering, FFmpeg compositing) exposed via CLI (argparse) and Web (FastAPI backend + React frontend). Each module is independent with clear interfaces — data flows linearly from input through processing to output.

**Tech Stack:** Python 3.10+, FastAPI, React/Vite/TypeScript, openai-whisper, librosa, Pillow, FFmpeg

---

## File Map

```
mvgenerate/
├── requirements.txt               # Python dependencies
├── config.py                      # Shared config, data models, constants
├── mvgenerate.py                  # CLI entry point (argparse)
├── run.py                         # One-command launcher (backend + frontend)
├── align/
│   ├── __init__.py
│   ├── lyrics_preprocessor.py     # Clean Suno-format lyrics
│   └── lyrics_aligner.py         # Whisper forced alignment
├── audio/
│   ├── __init__.py
│   └── analyzer.py               # librosa feature extraction
├── render/
│   ├── __init__.py
│   ├── base.py                   # Base renderer, frame pipeline
│   ├── themes/
│   │   ├── __init__.py
│   │   ├── neon_pulse.py         # Theme 1: Neon Pulse
│   │   ├── vinyl_minimal.py     # Theme 2: Vinyl Minimal
│   │   └── wave_groove.py       # Theme 3: Wave Groove
│   └── lyrics/
│       ├── __init__.py
│       ├── karaoke.py            # Lyrics style 1: KTV highlight
│       ├── fade.py               # Lyrics style 2: Fade in/out
│       └── word_fill.py          # Lyrics style 3: Word-by-word fill
├── output/
│   ├── __init__.py
│   └── composer.py               # FFmpeg video compositing
├── server/
│   ├── __init__.py
│   ├── app.py                    # FastAPI app
│   ├── routes.py                 # API endpoints
│   └── tasks.py                  # Background task manager
├── web/                          # React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── App.css
│       └── components/
│           ├── UploadArea.tsx
│           ├── ConfigPanel.tsx
│           ├── ProgressBar.tsx
│           └── ResultView.tsx
└── tests/
    ├── __init__.py
    ├── test_lyrics_preprocessor.py
    ├── test_audio_analyzer.py
    ├── test_renderer.py
    └── test_composer.py
```

---

### Task 1: Project Setup and Config

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `align/__init__.py`, `audio/__init__.py`, `render/__init__.py`, `render/themes/__init__.py`, `render/lyrics/__init__.py`, `output/__init__.py`, `server/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
faster-whisper==1.1.0
librosa==0.10.2
Pillow==11.1.0
numpy>=1.26,<2.0
soundfile==0.13.1
fastapi==0.115.6
uvicorn==0.34.0
python-multipart==0.0.20
```

- [ ] **Step 2: Create config.py with data models and constants**

```python
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


@dataclass
class GenerateConfig:
    audio_path: Path
    lyrics_path: Path
    cover_path: Path
    output_path: Path = Path("output.mp4")
    aspect: AspectRatio = AspectRatio.PORTRAIT
    theme: Theme = Theme.NEON
    lyrics_style: LyricsStyle = LyricsStyle.KARAOKE
    title: str = ""
    artist: str = ""
```

- [ ] **Step 3: Create all `__init__.py` files**

Run:
```bash
mkdir -p align audio render/themes render/lyrics output server tests
touch align/__init__.py audio/__init__.py render/__init__.py render/themes/__init__.py render/lyrics/__init__.py output/__init__.py server/__init__.py tests/__init__.py
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 5: Verify config imports**

Run: `python -c "from config import GenerateConfig, Theme, LyricsStyle, AspectRatio; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt config.py align/__init__.py audio/__init__.py render/__init__.py render/themes/__init__.py render/lyrics/__init__.py output/__init__.py server/__init__.py tests/__init__.py
git commit -m "feat: project setup with config and data models"
```

---

### Task 2: Lyrics Preprocessor

**Files:**
- Create: `align/lyrics_preprocessor.py`
- Create: `tests/test_lyrics_preprocessor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lyrics_preprocessor.py
from align.lyrics_preprocessor import preprocess_lyrics


def test_removes_section_markers():
    raw = "[Verse]\nfirst line\n[Chorus]\nsecond line"
    result = preprocess_lyrics(raw)
    assert result == ["first line", "second line"]


def test_removes_empty_lines():
    raw = "line one\n\n\nline two\n\n"
    result = preprocess_lyrics(raw)
    assert result == ["line one", "line two"]


def test_trims_whitespace():
    raw = "  hello world  \n  foo bar  "
    result = preprocess_lyrics(raw)
    assert result == ["hello world", "foo bar"]


def test_removes_various_suno_markers():
    raw = "[Intro]\n[Verse 1]\nlyric\n[Pre-Chorus]\n[Chorus]\nchorus line\n[Bridge]\n[Outro]"
    result = preprocess_lyrics(raw)
    assert result == ["lyric", "chorus line"]


def test_handles_brackets_in_lyrics():
    raw = "she said [softly] hello"
    result = preprocess_lyrics(raw)
    # Lines that are ONLY a bracket marker get removed; brackets embedded in lyrics stay
    assert result == ["she said [softly] hello"]


def test_empty_input():
    assert preprocess_lyrics("") == []
    assert preprocess_lyrics("\n\n\n") == []


def test_reads_from_file(tmp_path):
    from align.lyrics_preprocessor import preprocess_lyrics_file

    f = tmp_path / "lyrics.txt"
    f.write_text("[Verse]\nhello\n\nworld\n", encoding="utf-8")
    result = preprocess_lyrics_file(f)
    assert result == ["hello", "world"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lyrics_preprocessor.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement lyrics_preprocessor.py**

```python
# align/lyrics_preprocessor.py
import re
from pathlib import Path

# Matches lines that are ONLY a section marker like [Verse], [Chorus 2], [Pre-Chorus], etc.
_SECTION_MARKER = re.compile(r"^\[[\w\s\-]+\]$")


def preprocess_lyrics(raw_text: str) -> list[str]:
    """Clean raw lyrics text: remove section markers, blank lines, trim whitespace."""
    lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_MARKER.match(stripped):
            continue
        lines.append(stripped)
    return lines


def preprocess_lyrics_file(path: Path) -> list[str]:
    """Read a lyrics file and preprocess it."""
    text = path.read_text(encoding="utf-8")
    return preprocess_lyrics(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lyrics_preprocessor.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add align/lyrics_preprocessor.py tests/test_lyrics_preprocessor.py
git commit -m "feat: lyrics preprocessor with Suno format support"
```

---

### Task 3: Audio Analyzer

**Files:**
- Create: `audio/analyzer.py`
- Create: `tests/test_audio_analyzer.py`

- [ ] **Step 1: Write failing tests**

We need a test audio file. Generate a short sine wave for testing.

```python
# tests/test_audio_analyzer.py
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from audio.analyzer import analyze_audio
from config import FPS


@pytest.fixture
def sine_wav(tmp_path) -> Path:
    """Generate a 2-second 440Hz sine wave as a test audio file."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave with amplitude modulation to create volume variation
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, sr)
    return path


def test_analyze_returns_audio_features(sine_wav):
    features = analyze_audio(sine_wav)
    expected_frames = int(features.duration * FPS)
    # RMS array length should match expected frame count (allow +-1 for rounding)
    assert abs(len(features.rms) - expected_frames) <= 1
    assert abs(len(features.spectrum) - expected_frames) <= 1
    assert features.duration == pytest.approx(2.0, abs=0.1)


def test_rms_values_normalized(sine_wav):
    features = analyze_audio(sine_wav)
    assert all(0.0 <= v <= 1.0 for v in features.rms)


def test_spectrum_has_bands(sine_wav):
    features = analyze_audio(sine_wav)
    # Each frame should have multiple frequency bands
    assert len(features.spectrum[0]) >= 8


def test_beat_frames_are_valid_indices(sine_wav):
    features = analyze_audio(sine_wav)
    num_frames = len(features.rms)
    for bf in features.beat_frames:
        assert 0 <= bf < num_frames
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio_analyzer.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement audio/analyzer.py**

```python
# audio/analyzer.py
import numpy as np
import librosa
from pathlib import Path

from config import AudioFeatures, FPS

# Number of frequency bands for spectrum visualization
NUM_BANDS = 16


def analyze_audio(audio_path: Path) -> AudioFeatures:
    """Extract per-frame audio features for visualization."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    num_frames = int(duration * FPS)

    hop_length = sr // FPS  # samples per video frame

    # RMS energy per frame
    rms_raw = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    # Resample to exact frame count
    rms_resampled = np.interp(
        np.linspace(0, len(rms_raw) - 1, num_frames),
        np.arange(len(rms_raw)),
        rms_raw,
    )
    # Normalize to 0-1
    rms_max = rms_resampled.max() if rms_resampled.max() > 0 else 1.0
    rms = (rms_resampled / rms_max).tolist()

    # Mel spectrogram, reduced to NUM_BANDS
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NUM_BANDS, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize each band to 0-1
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_range = mel_max - mel_min if mel_max > mel_min else 1.0
    mel_norm = (mel_db - mel_min) / mel_range

    # Resample to exact frame count
    spectrum: list[list[float]] = []
    for i in range(num_frames):
        src_idx = int(i * mel_norm.shape[1] / num_frames)
        src_idx = min(src_idx, mel_norm.shape[1] - 1)
        spectrum.append(mel_norm[:, src_idx].tolist())

    # Beat detection
    tempo, beat_frame_indices = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    # Convert librosa frame indices to video frame indices
    beat_times = librosa.frames_to_time(beat_frame_indices, sr=sr, hop_length=hop_length)
    beat_frames = [int(t * FPS) for t in beat_times if int(t * FPS) < num_frames]

    return AudioFeatures(
        rms=rms,
        spectrum=spectrum,
        beat_frames=beat_frames,
        duration=duration,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio_analyzer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add audio/analyzer.py tests/test_audio_analyzer.py
git commit -m "feat: audio analyzer with RMS, spectrum, and beat detection"
```

---

### Task 4: Lyrics Aligner (Whisper)

**Files:**
- Create: `align/lyrics_aligner.py`

Note: Whisper alignment requires a real audio file with speech, so unit testing with synthetic audio is not reliable. We test this module with a manual integration test at the end. The interface is straightforward.

- [ ] **Step 1: Implement lyrics_aligner.py**

```python
# align/lyrics_aligner.py
from pathlib import Path
from faster_whisper import WhisperModel

from config import TimedLine, TimedWord


def align_lyrics(audio_path: Path, lyrics_lines: list[str], model_size: str = "large-v3") -> list[TimedLine]:
    """
    Align lyrics lines to audio using Whisper forced alignment.

    1. Transcribe the audio with Whisper (word-level timestamps).
    2. Match each provided lyrics line against the transcription.
    3. Return TimedLine objects with start/end times and word timestamps.
    """
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    segments, _info = model.transcribe(
        str(audio_path),
        language=None,  # auto-detect
        word_timestamps=True,
    )

    # Collect all transcribed words with timestamps
    transcribed_words: list[TimedWord] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                transcribed_words.append(TimedWord(
                    text=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))

    if not transcribed_words:
        # Fallback: evenly distribute lyrics across audio duration
        return _even_distribute(lyrics_lines, _info.duration)

    # Match lyrics lines to transcribed words using greedy alignment
    timed_lines = _match_lines_to_words(lyrics_lines, transcribed_words)
    return timed_lines


def _match_lines_to_words(lyrics_lines: list[str], words: list[TimedWord]) -> list[TimedLine]:
    """
    Greedy matching: for each lyrics line, find the best matching span
    in the transcribed words by character overlap.
    """
    timed_lines: list[TimedLine] = []
    word_idx = 0

    for line in lyrics_lines:
        line_chars = line.replace(" ", "").lower()
        if not line_chars:
            continue

        best_start_idx = word_idx
        best_score = 0
        best_end_idx = word_idx

        # Sliding window over remaining words
        for start in range(word_idx, len(words)):
            matched_chars = ""
            for end in range(start, min(start + len(line_chars) * 2, len(words))):
                matched_chars += words[end].text.replace(" ", "").lower()
                # Calculate overlap score
                score = _char_overlap(line_chars, matched_chars)
                if score > best_score:
                    best_score = score
                    best_start_idx = start
                    best_end_idx = end + 1
                # Early exit if perfect match
                if score >= len(line_chars):
                    break

        if best_score > 0 and best_start_idx < len(words):
            matched_words = words[best_start_idx:best_end_idx]
            timed_lines.append(TimedLine(
                text=line,
                start=matched_words[0].start,
                end=matched_words[-1].end,
                words=matched_words,
            ))
            word_idx = best_end_idx
        else:
            # No match found — will be filled in by gap filling later
            timed_lines.append(TimedLine(text=line, start=0.0, end=0.0, words=[]))

    # Fill gaps for unmatched lines
    _fill_gaps(timed_lines, words[-1].end if words else 0.0)

    return timed_lines


def _char_overlap(a: str, b: str) -> int:
    """Count matching characters between two strings (order-sensitive)."""
    matches = 0
    b_idx = 0
    for ch in a:
        while b_idx < len(b):
            if b[b_idx] == ch:
                matches += 1
                b_idx += 1
                break
            b_idx += 1
    return matches


def _fill_gaps(lines: list[TimedLine], total_duration: float) -> None:
    """Fill start/end times for lines that had no match, using surrounding lines."""
    for i, line in enumerate(lines):
        if line.start == 0.0 and line.end == 0.0:
            prev_end = lines[i - 1].end if i > 0 else 0.0
            next_start = total_duration
            for j in range(i + 1, len(lines)):
                if lines[j].start > 0:
                    next_start = lines[j].start
                    break
            gap = next_start - prev_end
            line.start = prev_end
            line.end = prev_end + gap


def _even_distribute(lyrics_lines: list[str], duration: float) -> list[TimedLine]:
    """Fallback: evenly space lyrics across the audio duration."""
    if not lyrics_lines:
        return []
    line_duration = duration / len(lyrics_lines)
    return [
        TimedLine(
            text=line,
            start=i * line_duration,
            end=(i + 1) * line_duration,
        )
        for i, line in enumerate(lyrics_lines)
    ]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from align.lyrics_aligner import align_lyrics; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add align/lyrics_aligner.py
git commit -m "feat: Whisper-based lyrics aligner with greedy matching"
```

---

### Task 5: Render Base and Frame Pipeline

**Files:**
- Create: `render/base.py`
- Create: `tests/test_renderer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_renderer.py
from PIL import Image
from config import (
    AudioFeatures, TimedLine, AspectRatio, Theme, LyricsStyle,
    RESOLUTIONS, FPS, COVER_DURATION, TRANSITION_DURATION, OUTRO_DURATION,
)
from render.base import Renderer


def _make_features(duration: float = 10.0) -> AudioFeatures:
    num_frames = int(duration * FPS)
    return AudioFeatures(
        rms=[0.5] * num_frames,
        spectrum=[[0.5] * 16 for _ in range(num_frames)],
        beat_frames=[i for i in range(0, num_frames, FPS)],  # beat every second
        duration=duration,
    )


def _make_lines() -> list[TimedLine]:
    return [
        TimedLine(text="first line", start=5.0, end=6.0),
        TimedLine(text="second line", start=6.5, end=7.5),
    ]


def test_renderer_produces_correct_size_frames(tmp_path):
    cover = Image.new("RGB", (500, 500), "red")
    cover_path = tmp_path / "cover.png"
    cover.save(cover_path)

    renderer = Renderer(
        cover_path=cover_path,
        aspect=AspectRatio.PORTRAIT,
        theme=Theme.NEON,
        lyrics_style=LyricsStyle.KARAOKE,
        title="Test Song",
        artist="Test Artist",
    )

    features = _make_features(10.0)
    lines = _make_lines()
    w, h = RESOLUTIONS[AspectRatio.PORTRAIT]

    # Render a single frame at t=0 (cover phase)
    frame = renderer.render_frame(0, 0.0, features, lines)
    assert frame.size == (w, h)

    # Render a frame during main playback
    frame2 = renderer.render_frame(int(6.0 * FPS), 6.0, features, lines)
    assert frame2.size == (w, h)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_renderer.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement render/base.py**

```python
# render/base.py
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import math

from config import (
    AspectRatio, Theme, LyricsStyle,
    AudioFeatures, TimedLine,
    RESOLUTIONS, FPS,
    COVER_DURATION, TRANSITION_DURATION, OUTRO_DURATION,
)


class Renderer:
    """Orchestrates per-frame rendering by compositing layers."""

    def __init__(
        self,
        cover_path: Path,
        aspect: AspectRatio,
        theme: Theme,
        lyrics_style: LyricsStyle,
        title: str = "",
        artist: str = "",
    ):
        self.width, self.height = RESOLUTIONS[aspect]
        self.theme_name = theme
        self.lyrics_style = lyrics_style
        self.title = title
        self.artist = artist

        # Load and prepare cover image
        self.cover_original = Image.open(cover_path).convert("RGB")

        # Prepare circular disc version of cover
        disc_size = min(self.width, self.height) // 3
        self.disc_size = disc_size
        cover_resized = self.cover_original.resize((disc_size, disc_size), Image.LANCZOS)
        # Create circular mask
        mask = Image.new("L", (disc_size, disc_size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, disc_size - 1, disc_size - 1), fill=255)
        self.cover_disc = cover_resized
        self.disc_mask = mask

        # Import theme and lyrics renderers
        self._theme = _get_theme(theme)
        self._lyrics_renderer = _get_lyrics_renderer(lyrics_style)

    def render_frame(
        self,
        frame_idx: int,
        time_s: float,
        features: AudioFeatures,
        lines: list[TimedLine],
    ) -> Image.Image:
        """Render a single frame at the given time."""
        total_duration = features.duration + COVER_DURATION + TRANSITION_DURATION + OUTRO_DURATION
        playback_start = COVER_DURATION + TRANSITION_DURATION

        # Create background
        frame = self._theme.draw_background(self.width, self.height, frame_idx, features)

        if time_s < COVER_DURATION:
            # Phase 1: Cover display
            progress = time_s / COVER_DURATION  # 0 -> 1 fade in
            self._draw_cover_phase(frame, progress)

        elif time_s < playback_start:
            # Phase 2: Transition (cover shrinks to disc)
            progress = (time_s - COVER_DURATION) / TRANSITION_DURATION
            self._draw_transition_phase(frame, progress, frame_idx, features)

        elif time_s < total_duration - OUTRO_DURATION:
            # Phase 3: Main playback
            audio_time = time_s - playback_start
            audio_frame = min(frame_idx, len(features.rms) - 1)
            self._draw_playback_phase(frame, audio_frame, audio_time, features, lines)

        else:
            # Phase 4: Outro fade out
            remaining = total_duration - time_s
            alpha = max(0.0, remaining / OUTRO_DURATION)
            audio_time = time_s - playback_start
            audio_frame = min(frame_idx, len(features.rms) - 1)
            self._draw_playback_phase(frame, audio_frame, audio_time, features, lines)
            # Apply fade out overlay
            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, int(255 * (1 - alpha))))
            frame.paste(Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB"))

        return frame

    def _draw_cover_phase(self, frame: Image.Image, progress: float) -> None:
        """Draw the cover display phase with fade-in."""
        # Center the cover image
        cover_display_size = min(self.width, self.height) // 2
        cover_resized = self.cover_original.resize(
            (cover_display_size, cover_display_size), Image.LANCZOS
        )
        x = (self.width - cover_display_size) // 2
        y = (self.height - cover_display_size) // 3

        # Fade in by blending
        if progress < 1.0:
            bg_crop = frame.crop((x, y, x + cover_display_size, y + cover_display_size))
            cover_resized = Image.blend(bg_crop, cover_resized, progress)

        frame.paste(cover_resized, (x, y))

        # Draw title and artist text below cover
        draw = ImageDraw.Draw(frame)
        if self.title:
            text_y = y + cover_display_size + 40
            self._theme.draw_title_text(draw, self.title, self.width, text_y, self.width)
        if self.artist:
            text_y = y + cover_display_size + 100
            self._theme.draw_artist_text(draw, self.artist, self.width, text_y, self.width)

    def _draw_transition_phase(
        self, frame: Image.Image, progress: float,
        frame_idx: int, features: AudioFeatures,
    ) -> None:
        """Animate cover shrinking into disc position."""
        cover_display_size = min(self.width, self.height) // 2
        current_size = int(cover_display_size - (cover_display_size - self.disc_size) * progress)

        cover_resized = self.cover_original.resize((current_size, current_size), Image.LANCZOS)
        # Transition to circular shape
        mask = Image.new("L", (current_size, current_size), 0)
        corner_radius = int(current_size * 0.5 * progress)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, current_size - 1, current_size - 1),
            radius=corner_radius, fill=255,
        )

        disc_x = (self.width - current_size) // 2
        # Animate from cover position to disc position
        start_y = (self.height - cover_display_size) // 3
        end_y = self.height // 4
        disc_y = int(start_y + (end_y - start_y) * progress)

        frame.paste(cover_resized, (disc_x, disc_y), mask)

    def _draw_playback_phase(
        self, frame: Image.Image,
        audio_frame: int, audio_time: float,
        features: AudioFeatures, lines: list[TimedLine],
    ) -> None:
        """Draw the main playback: disc + visualizer + lyrics."""
        disc_x = (self.width - self.disc_size) // 2
        disc_y = self.height // 4

        # Rotate disc based on time
        angle = (audio_time * 45) % 360  # 45 degrees per second
        rotated = self.cover_disc.rotate(-angle, resample=Image.BICUBIC, expand=False)
        frame.paste(rotated, (disc_x, disc_y), self.disc_mask)

        # Draw disc ring/effects via theme
        self._theme.draw_disc_effects(frame, disc_x, disc_y, self.disc_size, audio_frame, features)

        # Draw audio visualizer via theme
        self._theme.draw_visualizer(frame, audio_frame, features, self.width, self.height)

        # Draw lyrics
        self._lyrics_renderer.draw(frame, audio_time, lines, self._theme, self.width, self.height)

    def total_frames(self, audio_duration: float) -> int:
        """Total number of frames for the full video."""
        total = audio_duration + COVER_DURATION + TRANSITION_DURATION + OUTRO_DURATION
        return int(total * FPS)


def _get_theme(theme: Theme):
    if theme == Theme.NEON:
        from render.themes.neon_pulse import NeonPulseTheme
        return NeonPulseTheme()
    elif theme == Theme.VINYL:
        from render.themes.vinyl_minimal import VinylMinimalTheme
        return VinylMinimalTheme()
    elif theme == Theme.WAVE:
        from render.themes.wave_groove import WaveGrooveTheme
        return WaveGrooveTheme()


def _get_lyrics_renderer(style: LyricsStyle):
    if style == LyricsStyle.KARAOKE:
        from render.lyrics.karaoke import KaraokeLyrics
        return KaraokeLyrics()
    elif style == LyricsStyle.FADE:
        from render.lyrics.fade import FadeLyrics
        return FadeLyrics()
    elif style == LyricsStyle.WORD_FILL:
        from render.lyrics.word_fill import WordFillLyrics
        return WordFillLyrics()
```

- [ ] **Step 4: Create stub theme and lyrics renderers so base.py can instantiate**

We need minimal stubs for the test to pass. These will be replaced in subsequent tasks.

```python
# render/themes/neon_pulse.py
from PIL import Image, ImageDraw
from config import AudioFeatures


class NeonPulseTheme:
    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        """Deep blue-to-purple gradient background."""
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            ratio = y / h
            r = int(10 + 20 * ratio)
            g = int(5 + 10 * (1 - ratio))
            b = int(40 + 60 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        return img

    def draw_disc_effects(self, frame, x, y, size, audio_frame, features):
        pass

    def draw_visualizer(self, frame, audio_frame, features, w, h):
        pass

    def draw_title_text(self, draw, text, canvas_w, y, max_w):
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt")

    def draw_artist_text(self, draw, text, canvas_w, y, max_w):
        draw.text((canvas_w // 2, y), text, fill=(200, 200, 200), anchor="mt")

    def get_lyrics_color(self):
        return (255, 255, 255)

    def get_lyrics_highlight_color(self):
        return (0, 255, 255)

    def get_lyrics_dim_color(self):
        return (100, 100, 100)
```

```python
# render/lyrics/karaoke.py
from PIL import Image, ImageDraw
from config import TimedLine


class KaraokeLyrics:
    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        """Draw KTV-style lyrics with current line highlighted."""
        draw = ImageDraw.Draw(frame)
        # Find current line
        current_idx = 0
        for i, line in enumerate(lines):
            if line.start <= time_s <= line.end:
                current_idx = i
                break
            if line.start > time_s:
                current_idx = max(0, i - 1)
                break

        # Display surrounding lines
        visible_range = 2
        lyrics_y = int(h * 0.65)
        line_height = 50

        for offset in range(-visible_range, visible_range + 1):
            idx = current_idx + offset
            if 0 <= idx < len(lines):
                y = lyrics_y + offset * line_height
                if offset == 0:
                    color = theme.get_lyrics_highlight_color()
                else:
                    color = theme.get_lyrics_dim_color()
                draw.text((w // 2, y), lines[idx].text, fill=color, anchor="mt")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_renderer.py -v`
Expected: All 1 test PASS (produces correct size frames)

- [ ] **Step 6: Commit**

```bash
git add render/base.py render/themes/neon_pulse.py render/lyrics/karaoke.py tests/test_renderer.py
git commit -m "feat: render pipeline with base renderer, neon stub, karaoke stub"
```

---

### Task 6: Neon Pulse Theme (Full Implementation)

**Files:**
- Modify: `render/themes/neon_pulse.py`

- [ ] **Step 1: Implement full Neon Pulse theme**

```python
# render/themes/neon_pulse.py
import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from config import AudioFeatures


class NeonPulseTheme:
    """Cyberpunk/neon themed visuals with glowing effects."""

    def __init__(self):
        self._particles: list[dict] | None = None

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        """Deep blue-to-purple gradient with floating particles."""
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)

        # Gradient background
        for y in range(h):
            ratio = y / h
            r = int(8 + 25 * ratio)
            g = int(2 + 8 * (1 - ratio))
            b = int(35 + 70 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Floating particles
        if self._particles is None:
            self._particles = [
                {
                    "x": random.randint(0, w),
                    "y": random.randint(0, h),
                    "speed": random.uniform(0.3, 1.5),
                    "size": random.randint(1, 3),
                    "phase": random.uniform(0, math.pi * 2),
                }
                for _ in range(40)
            ]

        for p in self._particles:
            px = int(p["x"] + math.sin(frame_idx * 0.02 + p["phase"]) * 20)
            py = int((p["y"] - frame_idx * p["speed"]) % h)
            alpha = int(80 + 60 * math.sin(frame_idx * 0.05 + p["phase"]))
            draw.ellipse(
                (px - p["size"], py - p["size"], px + p["size"], py + p["size"]),
                fill=(100, 200, 255, alpha) if img.mode == "RGBA" else (100, 200, 255),
            )

        return img

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        """Draw neon glow ring around the disc that pulses with beat."""
        draw = ImageDraw.Draw(frame)
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 2

        # Pulse intensity from RMS
        rms = features.rms[audio_frame] if audio_frame < len(features.rms) else 0.5
        is_beat = audio_frame in features.beat_frames

        glow_width = int(4 + 8 * rms)
        if is_beat:
            glow_width += 6

        # Draw multiple concentric rings for glow effect
        for i in range(glow_width, 0, -1):
            alpha_ratio = (glow_width - i) / glow_width
            # Cycle colors: cyan -> pink -> purple
            phase = (audio_frame * 0.03) % (math.pi * 2)
            r = int(80 + 175 * max(0, math.sin(phase)))
            g = int(200 * max(0, math.sin(phase + math.pi * 2 / 3)))
            b = int(150 + 105 * max(0, math.sin(phase + math.pi * 4 / 3)))
            color = (r, g, b)

            ring_r = radius + 5 + i
            draw.ellipse(
                (center_x - ring_r, center_y - ring_r, center_x + ring_r, center_y + ring_r),
                outline=color, width=1,
            )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        """Draw circular spectrum bars around the disc area."""
        draw = ImageDraw.Draw(frame)
        center_x = w // 2
        center_y = h // 4 + (min(w, h) // 3) // 2  # center of disc

        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        num_bars = len(spectrum)
        base_radius = min(w, h) // 3 // 2 + 30  # just outside disc ring

        for i, val in enumerate(spectrum):
            angle = (i / num_bars) * math.pi * 2 - math.pi / 2
            bar_length = int(20 + 80 * val)

            x1 = center_x + int(base_radius * math.cos(angle))
            y1 = center_y + int(base_radius * math.sin(angle))
            x2 = center_x + int((base_radius + bar_length) * math.cos(angle))
            y2 = center_y + int((base_radius + bar_length) * math.sin(angle))

            # Color cycles with beat
            phase = (audio_frame * 0.03 + i * 0.2) % (math.pi * 2)
            r = int(80 + 175 * max(0, math.sin(phase)))
            g = int(200 * max(0, math.sin(phase + math.pi * 2 / 3)))
            b = int(150 + 105 * max(0, math.sin(phase + math.pi * 4 / 3)))

            draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=3)

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt", font=font)

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(180, 180, 200), anchor="mt", font=font)

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (255, 255, 255)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (0, 255, 255)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (100, 100, 120)
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/test_renderer.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add render/themes/neon_pulse.py
git commit -m "feat: full neon pulse theme with particles, glow ring, circular spectrum"
```

---

### Task 7: Vinyl Minimal and Wave Groove Themes

**Files:**
- Create: `render/themes/vinyl_minimal.py`
- Create: `render/themes/wave_groove.py`

- [ ] **Step 1: Implement Vinyl Minimal theme**

```python
# render/themes/vinyl_minimal.py
import math
from PIL import Image, ImageDraw, ImageFont
from config import AudioFeatures


class VinylMinimalTheme:
    """Clean, minimal theme with vinyl record aesthetic."""

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        # Soft warm gradient (cream to light gray)
        for y in range(h):
            ratio = y / h
            r = int(240 - 20 * ratio)
            g = int(235 - 25 * ratio)
            b = int(225 - 20 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        return img

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        """Draw vinyl record grooves and tonearm."""
        draw = ImageDraw.Draw(frame)
        cx = x + size // 2
        cy = y + size // 2
        radius = size // 2

        # Vinyl grooves (concentric dark circles)
        for i in range(3, radius, 8):
            draw.ellipse(
                (cx - radius - i, cy - radius - i, cx + radius + i, cy + radius + i),
                outline=(60, 55, 50), width=1,
            )

        # Tonearm - simple line from top-right
        arm_start_x = x + size + 20
        arm_start_y = y - 10
        arm_angle = 0.7 + 0.1 * math.sin(audio_frame * 0.01)
        arm_length = size // 2 + 30
        arm_end_x = int(arm_start_x - arm_length * math.sin(arm_angle))
        arm_end_y = int(arm_start_y + arm_length * math.cos(arm_angle))
        draw.line([(arm_start_x, arm_start_y), (arm_end_x, arm_end_y)], fill=(80, 75, 70), width=3)
        # Pivot dot
        draw.ellipse(
            (arm_start_x - 5, arm_start_y - 5, arm_start_x + 5, arm_start_y + 5),
            fill=(80, 75, 70),
        )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        """Minimal bar waveform at the bottom."""
        draw = ImageDraw.Draw(frame)
        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        num_bars = len(spectrum)
        bar_width = max(2, w // (num_bars * 3))
        total_w = num_bars * bar_width * 2
        start_x = (w - total_w) // 2
        base_y = int(h * 0.88)

        for i, val in enumerate(spectrum):
            bar_h = int(5 + 60 * val)
            bx = start_x + i * bar_width * 2
            draw.rectangle(
                (bx, base_y - bar_h, bx + bar_width, base_y),
                fill=(60, 55, 50),
            )

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(40, 35, 30), anchor="mt", font=font)

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(100, 95, 90), anchor="mt", font=font)

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (40, 35, 30)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (20, 15, 10)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (160, 155, 150)
```

- [ ] **Step 2: Implement Wave Groove theme**

```python
# render/themes/wave_groove.py
import math
from PIL import Image, ImageDraw, ImageFont
from config import AudioFeatures


class WaveGrooveTheme:
    """Dynamic theme with flowing wave textures and breathing effects."""

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)

        # Dark background with slow-moving wave texture
        for y in range(h):
            wave = math.sin(y * 0.01 + frame_idx * 0.02) * 15
            ratio = y / h
            r = int(15 + 10 * ratio + wave)
            g = int(12 + 15 * ratio)
            b = int(25 + 20 * ratio + wave * 0.5)
            r = max(0, min(255, r))
            b = max(0, min(255, b))
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Horizontal wave lines
        rms = 0.5
        if frame_idx < len(features.rms):
            rms = features.rms[frame_idx]

        for wave_y_base in range(0, h, 120):
            points = []
            for x in range(0, w, 4):
                y_offset = math.sin(x * 0.01 + frame_idx * 0.03 + wave_y_base * 0.1) * (10 + 20 * rms)
                points.append((x, wave_y_base + y_offset))
            if len(points) > 1:
                draw.line(points, fill=(40, 50, 80), width=1)

        return img

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        """Breathing scale effect — disc subtly pulses with beat."""
        # The breathing effect is handled in base.py by varying the disc render size
        # Here we just draw a subtle ring
        draw = ImageDraw.Draw(frame)
        cx = x + size // 2
        cy = y + size // 2
        rms = features.rms[audio_frame] if audio_frame < len(features.rms) else 0.5
        ring_r = size // 2 + 5 + int(8 * rms)
        draw.ellipse(
            (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
            outline=(60, 80, 140), width=2,
        )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        """Circular wave lines around disc area."""
        draw = ImageDraw.Draw(frame)
        cx = w // 2
        cy = h // 4 + (min(w, h) // 3) // 2

        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        base_r = min(w, h) // 3 // 2 + 20

        # Draw wavy ring
        points = []
        num_points = 64
        for i in range(num_points + 1):
            angle = (i / num_points) * math.pi * 2
            band_idx = int((i / num_points) * len(spectrum)) % len(spectrum)
            wave = spectrum[band_idx] * 40
            r = base_r + wave + math.sin(angle * 3 + audio_frame * 0.05) * 10
            px = cx + int(r * math.cos(angle))
            py = cy + int(r * math.sin(angle))
            points.append((px, py))

        if len(points) > 1:
            draw.line(points, fill=(80, 120, 200), width=2)

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt", font=font)

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(160, 180, 220), anchor="mt", font=font)

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (255, 255, 255)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (120, 180, 255)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (80, 90, 110)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_renderer.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add render/themes/vinyl_minimal.py render/themes/wave_groove.py
git commit -m "feat: vinyl minimal and wave groove themes"
```

---

### Task 8: Karaoke, Fade, and Word-Fill Lyrics Renderers

**Files:**
- Modify: `render/lyrics/karaoke.py` (full implementation)
- Create: `render/lyrics/fade.py`
- Create: `render/lyrics/word_fill.py`

- [ ] **Step 1: Full karaoke lyrics renderer**

```python
# render/lyrics/karaoke.py
from PIL import Image, ImageDraw, ImageFont
from config import TimedLine


class KaraokeLyrics:
    """KTV-style lyrics: show 3-5 lines, highlight current, smooth scroll."""

    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        if not lines:
            return

        draw = ImageDraw.Draw(frame)

        try:
            font_highlight = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 40)
            font_normal = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
        except OSError:
            font_highlight = ImageFont.load_default()
            font_normal = font_highlight

        # Find current line index
        current_idx = 0
        for i, line in enumerate(lines):
            if line.start <= time_s:
                current_idx = i
            if line.start > time_s:
                break

        # Display window: 2 lines above and below current
        visible_range = 2
        lyrics_area_y = int(h * 0.62)
        line_spacing = 55

        for offset in range(-visible_range, visible_range + 1):
            idx = current_idx + offset
            if idx < 0 or idx >= len(lines):
                continue

            y = lyrics_area_y + offset * line_spacing

            # Smooth scroll: interpolate y based on progress within current line
            if current_idx < len(lines):
                cur = lines[current_idx]
                if cur.end > cur.start:
                    progress = min(1.0, (time_s - cur.start) / (cur.end - cur.start))
                    y -= int(progress * line_spacing * 0.3)

            if offset == 0:
                color = theme.get_lyrics_highlight_color()
                draw.text((w // 2, y), lines[idx].text, fill=color, anchor="mt", font=font_highlight)
            else:
                # Fade dim lines by distance
                dim = theme.get_lyrics_dim_color()
                dist = abs(offset)
                fade = max(0.3, 1.0 - dist * 0.25)
                color = tuple(int(c * fade) for c in dim)
                draw.text((w // 2, y), lines[idx].text, fill=color, anchor="mt", font=font_normal)
```

- [ ] **Step 2: Implement fade lyrics renderer**

```python
# render/lyrics/fade.py
from PIL import Image, ImageDraw, ImageFont
from config import TimedLine


class FadeLyrics:
    """Single line display with fade in/out transitions."""

    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        if not lines:
            return

        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 38)
        except OSError:
            font = ImageFont.load_default()

        lyrics_y = int(h * 0.68)

        # Find current and next lines
        current = None
        next_line = None
        for i, line in enumerate(lines):
            if line.start <= time_s <= line.end:
                current = line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                break

        if current is None:
            return

        # Calculate fade alpha
        fade_duration = 0.3  # seconds for fade transition
        line_duration = current.end - current.start

        if time_s - current.start < fade_duration:
            # Fading in
            alpha = (time_s - current.start) / fade_duration
        elif current.end - time_s < fade_duration:
            # Fading out
            alpha = (current.end - time_s) / fade_duration
        else:
            alpha = 1.0

        alpha = max(0.0, min(1.0, alpha))

        highlight = theme.get_lyrics_highlight_color()
        color = tuple(int(c * alpha) for c in highlight)
        draw.text((w // 2, lyrics_y), current.text, fill=color, anchor="mt", font=font)
```

- [ ] **Step 3: Implement word-fill lyrics renderer**

```python
# render/lyrics/word_fill.py
from PIL import Image, ImageDraw, ImageFont
from config import TimedLine


class WordFillLyrics:
    """Apple Music style: words fill with color as they're sung."""

    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        if not lines:
            return

        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 38)
        except OSError:
            font = ImageFont.load_default()

        lyrics_y = int(h * 0.68)

        # Find current line
        current = None
        for line in lines:
            if line.start <= time_s <= line.end:
                current = line
                break

        if current is None:
            return

        highlight_color = theme.get_lyrics_highlight_color()
        dim_color = theme.get_lyrics_dim_color()

        if not current.words:
            # No word-level timestamps — fallback: proportional fill
            progress = (time_s - current.start) / max(0.01, current.end - current.start)
            text = current.text
            fill_chars = int(len(text) * progress)

            # Draw filled portion
            filled = text[:fill_chars]
            remaining = text[fill_chars:]

            # Calculate position so text is centered
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            start_x = (w - text_w) // 2

            if filled:
                draw.text((start_x, lyrics_y), filled, fill=highlight_color, anchor="lt", font=font)
                filled_w = font.getbbox(filled)[2] - font.getbbox(filled)[0]
            else:
                filled_w = 0

            if remaining:
                draw.text((start_x + filled_w, lyrics_y), remaining, fill=dim_color, anchor="lt", font=font)
        else:
            # Word-level fill
            text = current.text
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            start_x = (w - text_w) // 2

            # Build character-level coloring based on word timestamps
            x_cursor = start_x
            for word in current.words:
                word_text = word.text
                if time_s >= word.end:
                    color = highlight_color
                elif time_s >= word.start:
                    # Partially filled
                    progress = (time_s - word.start) / max(0.01, word.end - word.start)
                    color = tuple(
                        int(d + (h_ - d) * progress)
                        for d, h_ in zip(dim_color, highlight_color)
                    )
                else:
                    color = dim_color

                draw.text((x_cursor, lyrics_y), word_text, fill=color, anchor="lt", font=font)
                word_bbox = font.getbbox(word_text)
                x_cursor += word_bbox[2] - word_bbox[0]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add render/lyrics/karaoke.py render/lyrics/fade.py render/lyrics/word_fill.py
git commit -m "feat: karaoke, fade, and word-fill lyrics renderers"
```

---

### Task 9: FFmpeg Video Composer

**Files:**
- Create: `output/composer.py`
- Create: `tests/test_composer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_composer.py
import subprocess
from pathlib import Path
from PIL import Image

from output.composer import compose_video


def test_compose_creates_mp4(tmp_path):
    """Test that composer creates a valid MP4 from frames + audio."""
    # Create dummy frames
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(30):  # 1 second at 30fps
        img = Image.new("RGB", (320, 240), (i * 8, 100, 200))
        img.save(frames_dir / f"frame_{i:06d}.png")

    # Create a silent audio file (1 second)
    audio_path = tmp_path / "silence.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", "1", str(audio_path),
    ], capture_output=True)

    output_path = tmp_path / "output.mp4"
    compose_video(frames_dir, audio_path, output_path, fps=30)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_composer.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement output/composer.py**

```python
# output/composer.py
import subprocess
from pathlib import Path


def compose_video(
    frames_dir: Path,
    audio_path: Path,
    output_path: Path,
    fps: int = 30,
) -> None:
    """Combine frame images + audio into an MP4 video using FFmpeg."""
    frame_pattern = str(frames_dir / "frame_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_composer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add output/composer.py tests/test_composer.py
git commit -m "feat: FFmpeg video composer"
```

---

### Task 10: CLI Entry Point

**Files:**
- Create: `mvgenerate.py`

- [ ] **Step 1: Implement CLI**

```python
#!/usr/bin/env python3
# mvgenerate.py
"""MV Generate — CLI entry point."""
import argparse
import sys
import tempfile
from pathlib import Path

from config import (
    AspectRatio, Theme, LyricsStyle, GenerateConfig,
    FPS, COVER_DURATION, TRANSITION_DURATION, OUTRO_DURATION,
)
from align.lyrics_preprocessor import preprocess_lyrics_file
from align.lyrics_aligner import align_lyrics
from audio.analyzer import analyze_audio
from render.base import Renderer
from output.composer import compose_video


def generate(config: GenerateConfig, progress_callback=None) -> None:
    """Run the full generation pipeline."""

    def report(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)
        else:
            print(f"[{pct:.0%}] {msg}")

    # Step 1: Preprocess lyrics
    report("Preprocessing lyrics...", 0.0)
    lyrics_lines = preprocess_lyrics_file(config.lyrics_path)
    if not lyrics_lines:
        raise ValueError("No lyrics found after preprocessing.")

    # Step 2: Align lyrics with audio
    report("Aligning lyrics with audio (this may take a while)...", 0.05)
    timed_lines = align_lyrics(config.audio_path, lyrics_lines)

    # Step 3: Analyze audio
    report("Analyzing audio...", 0.20)
    features = analyze_audio(config.audio_path)

    # Step 4: Render frames
    report("Initializing renderer...", 0.25)
    renderer = Renderer(
        cover_path=config.cover_path,
        aspect=config.aspect,
        theme=config.theme,
        lyrics_style=config.lyrics_style,
        title=config.title,
        artist=config.artist,
    )

    total_frames = renderer.total_frames(features.duration)

    with tempfile.TemporaryDirectory(prefix="mvgen_") as tmp_dir:
        frames_dir = Path(tmp_dir) / "frames"
        frames_dir.mkdir()

        for i in range(total_frames):
            time_s = i / FPS
            frame = renderer.render_frame(i, time_s, features, timed_lines)
            frame.save(frames_dir / f"frame_{i:06d}.png")

            if i % FPS == 0:  # report every second
                pct = 0.25 + 0.65 * (i / total_frames)
                report(f"Rendering frame {i}/{total_frames}...", pct)

        # Step 5: Compose video
        report("Composing final video...", 0.90)
        compose_video(frames_dir, config.audio_path, config.output_path, FPS)

    report(f"Done! Video saved to {config.output_path}", 1.0)


def main():
    parser = argparse.ArgumentParser(description="Generate a lyric music video from MP3 + lyrics + cover image.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to MP3 audio file")
    parser.add_argument("--lyrics", type=Path, required=True, help="Path to lyrics text file")
    parser.add_argument("--cover", type=Path, required=True, help="Path to cover image (JPG/PNG)")
    parser.add_argument("--output", type=Path, default=Path("output.mp4"), help="Output MP4 path (default: output.mp4)")
    parser.add_argument("--aspect", choices=["9:16", "16:9"], default="9:16", help="Aspect ratio (default: 9:16)")
    parser.add_argument("--theme", choices=["neon", "vinyl", "wave"], default="neon", help="Visual theme (default: neon)")
    parser.add_argument("--lyrics-style", choices=["karaoke", "fade", "word-fill"], default="karaoke", help="Lyrics display style (default: karaoke)")
    parser.add_argument("--title", default="", help="Song title (shown on cover)")
    parser.add_argument("--artist", default="", help="Artist name (shown on cover)")

    args = parser.parse_args()

    # Validate input files exist
    for path, name in [(args.audio, "Audio"), (args.lyrics, "Lyrics"), (args.cover, "Cover")]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    config = GenerateConfig(
        audio_path=args.audio,
        lyrics_path=args.lyrics,
        cover_path=args.cover,
        output_path=args.output,
        aspect=AspectRatio(args.aspect),
        theme=Theme(args.theme),
        lyrics_style=LyricsStyle(args.lyrics_style),
        title=args.title,
        artist=args.artist,
    )

    generate(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python mvgenerate.py --help`
Expected: Shows help text with all options

- [ ] **Step 3: Commit**

```bash
git add mvgenerate.py
git commit -m "feat: CLI entry point with full generation pipeline"
```

---

### Task 11: FastAPI Backend

**Files:**
- Create: `server/app.py`
- Create: `server/routes.py`
- Create: `server/tasks.py`

- [ ] **Step 1: Implement task manager**

```python
# server/tasks.py
import uuid
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result_path: str = ""
    error: str = ""


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = TaskInfo(task_id=task_id)
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for key, value in kwargs.items():
                    setattr(task, key, value)

    def run_in_background(self, task_id: str, fn: Callable, *args, **kwargs) -> None:
        def wrapper():
            self.update_task(task_id, status=TaskStatus.RUNNING)
            try:
                fn(*args, **kwargs)
                self.update_task(task_id, status=TaskStatus.COMPLETED, progress=1.0, message="Done")
            except Exception as e:
                self.update_task(task_id, status=TaskStatus.FAILED, error=str(e))

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()


task_manager = TaskManager()
```

- [ ] **Step 2: Implement API routes**

```python
# server/routes.py
import json
import shutil
import asyncio
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from config import AspectRatio, Theme, LyricsStyle, GenerateConfig
from server.tasks import task_manager, TaskStatus
from mvgenerate import generate

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an MP3, lyrics txt, or cover image. Returns a file ID."""
    file_id = task_manager.create_task()[:8]
    ext = Path(file.filename).suffix if file.filename else ""
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "filename": file.filename, "path": str(save_path)}


@router.post("/generate")
async def generate_video(
    audio_path: str = Form(...),
    lyrics_path: str = Form(...),
    cover_path: str = Form(...),
    aspect: str = Form("9:16"),
    theme: str = Form("neon"),
    lyrics_style: str = Form("karaoke"),
    title: str = Form(""),
    artist: str = Form(""),
):
    """Start video generation. Returns a task ID for progress tracking."""
    # Validate files exist
    for p, name in [(audio_path, "audio"), (lyrics_path, "lyrics"), (cover_path, "cover")]:
        if not Path(p).exists():
            raise HTTPException(400, f"{name} file not found: {p}")

    task_id = task_manager.create_task()
    output_path = OUTPUT_DIR / f"{task_id}.mp4"

    config = GenerateConfig(
        audio_path=Path(audio_path),
        lyrics_path=Path(lyrics_path),
        cover_path=Path(cover_path),
        output_path=output_path,
        aspect=AspectRatio(aspect),
        theme=Theme(theme),
        lyrics_style=LyricsStyle(lyrics_style),
        title=title,
        artist=artist,
    )

    def progress_callback(msg: str, pct: float):
        task_manager.update_task(task_id, message=msg, progress=pct)

    def run():
        generate(config, progress_callback=progress_callback)
        task_manager.update_task(task_id, result_path=str(output_path))

    task_manager.run_in_background(task_id, run)
    return {"task_id": task_id}


@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """SSE endpoint for real-time progress updates."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    async def event_stream():
        while True:
            task = task_manager.get_task(task_id)
            if task is None:
                break

            data = {
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
            }

            if task.status == TaskStatus.COMPLETED:
                data["result_path"] = task.result_path
                yield f"data: {json.dumps(data)}\n\n"
                break
            elif task.status == TaskStatus.FAILED:
                data["error"] = task.error
                yield f"data: {json.dumps(data)}\n\n"
                break

            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    """Download the generated video."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(400, f"Task not ready: {task.status.value}")
    if not Path(task.result_path).exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        task.result_path,
        media_type="video/mp4",
        filename=f"mv_{task_id}.mp4",
    )
```

- [ ] **Step 3: Implement FastAPI app**

```python
# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.routes import router

app = FastAPI(title="MV Generate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve frontend static files if built
frontend_dist = Path("web/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
```

- [ ] **Step 4: Verify server starts**

Run: `python -c "from server.app import app; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add server/app.py server/routes.py server/tasks.py
git commit -m "feat: FastAPI backend with upload, generate, progress SSE, and download endpoints"
```

---

### Task 12: React Frontend Setup

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/App.css`

- [ ] **Step 1: Create package.json**

```json
{
  "name": "mvgenerate-web",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 4: Create index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MV Generate</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Create main.tsx**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './App.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 6: Create App.css**

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0a0a1a;
  color: #e0e0e0;
  min-height: 100vh;
}

.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 40px 20px;
}

.app h1 {
  text-align: center;
  font-size: 2rem;
  margin-bottom: 40px;
  background: linear-gradient(135deg, #00ffff, #ff00ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.section {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 24px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.section h2 {
  font-size: 1rem;
  margin-bottom: 16px;
  color: #aaa;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.upload-zone {
  border: 2px dashed rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s;
  margin-bottom: 12px;
}

.upload-zone:hover {
  border-color: #00ffff;
}

.upload-zone.uploaded {
  border-color: #00ff88;
  border-style: solid;
}

.upload-zone p {
  color: #888;
  font-size: 0.9rem;
}

.upload-zone .filename {
  color: #00ff88;
  font-size: 0.85rem;
  margin-top: 4px;
}

.cover-preview {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  margin-top: 8px;
}

.radio-group {
  margin-bottom: 16px;
}

.radio-group label {
  display: block;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
  font-size: 0.95rem;
}

.radio-group label:hover {
  background: rgba(255, 255, 255, 0.08);
}

.radio-group input[type="radio"] {
  margin-right: 8px;
}

.text-inputs {
  margin-bottom: 24px;
}

.text-inputs input {
  width: 100%;
  padding: 10px 14px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 6px;
  color: white;
  font-size: 0.95rem;
}

.text-inputs input::placeholder {
  color: #666;
}

.generate-btn {
  display: block;
  width: 100%;
  padding: 14px;
  background: linear-gradient(135deg, #00cccc, #8800ff);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
  margin-bottom: 24px;
}

.generate-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.progress-section {
  margin-bottom: 24px;
}

.progress-bar-bg {
  width: 100%;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ffff, #8800ff);
  border-radius: 4px;
  transition: width 0.3s;
}

.progress-text {
  font-size: 0.85rem;
  color: #888;
}

.result-section video {
  width: 100%;
  max-height: 500px;
  border-radius: 8px;
  margin-bottom: 12px;
}

.download-btn {
  display: inline-block;
  padding: 10px 24px;
  background: #00cc88;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
  text-decoration: none;
}

@media (max-width: 640px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 7: Create App.tsx (shell — components in next task)**

```tsx
import { useState } from 'react'
import UploadArea from './components/UploadArea'
import ConfigPanel from './components/ConfigPanel'
import ProgressBar from './components/ProgressBar'
import ResultView from './components/ResultView'

interface UploadedFiles {
  audio?: { path: string; name: string }
  lyrics?: { path: string; name: string }
  cover?: { path: string; name: string; previewUrl?: string }
}

interface Config {
  aspect: string
  theme: string
  lyricsStyle: string
}

export default function App() {
  const [files, setFiles] = useState<UploadedFiles>({})
  const [config, setConfig] = useState<Config>({
    aspect: '9:16',
    theme: 'neon',
    lyricsStyle: 'karaoke',
  })
  const [title, setTitle] = useState('')
  const [artist, setArtist] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)

  const canGenerate = files.audio && files.lyrics && files.cover && !isGenerating

  async function handleGenerate() {
    if (!files.audio || !files.lyrics || !files.cover) return

    setIsGenerating(true)
    setProgress(0)
    setStatusMsg('Starting...')
    setResultUrl(null)

    const formData = new FormData()
    formData.append('audio_path', files.audio.path)
    formData.append('lyrics_path', files.lyrics.path)
    formData.append('cover_path', files.cover.path)
    formData.append('aspect', config.aspect)
    formData.append('theme', config.theme)
    formData.append('lyrics_style', config.lyricsStyle)
    formData.append('title', title)
    formData.append('artist', artist)

    const res = await fetch('/api/generate', { method: 'POST', body: formData })
    const { task_id } = await res.json()
    setTaskId(task_id)

    // Listen for progress via SSE
    const evtSource = new EventSource(`/api/progress/${task_id}`)
    evtSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setProgress(data.progress)
      setStatusMsg(data.message)

      if (data.status === 'completed') {
        setResultUrl(`/api/result/${task_id}`)
        setIsGenerating(false)
        evtSource.close()
      } else if (data.status === 'failed') {
        setStatusMsg(`Error: ${data.error}`)
        setIsGenerating(false)
        evtSource.close()
      }
    }
  }

  return (
    <div className="app">
      <h1>MV Generate</h1>

      <div className="main-grid">
        <UploadArea files={files} onFilesChange={setFiles} />
        <ConfigPanel config={config} onConfigChange={setConfig} />
      </div>

      <div className="text-inputs section">
        <input
          placeholder="Song title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <input
          placeholder="Artist name"
          value={artist}
          onChange={(e) => setArtist(e.target.value)}
        />
      </div>

      <button
        className="generate-btn"
        disabled={!canGenerate}
        onClick={handleGenerate}
      >
        Generate Video
      </button>

      {isGenerating && (
        <ProgressBar progress={progress} message={statusMsg} />
      )}

      {resultUrl && (
        <ResultView videoUrl={resultUrl} taskId={taskId!} />
      )}
    </div>
  )
}
```

- [ ] **Step 8: Install npm deps and verify build**

Run:
```bash
cd web && npm install
```
Expected: `node_modules` created

- [ ] **Step 9: Commit**

```bash
git add web/package.json web/tsconfig.json web/vite.config.ts web/index.html web/src/main.tsx web/src/App.tsx web/src/App.css
git commit -m "feat: React frontend setup with App shell and styles"
```

---

### Task 13: React Frontend Components

**Files:**
- Create: `web/src/components/UploadArea.tsx`
- Create: `web/src/components/ConfigPanel.tsx`
- Create: `web/src/components/ProgressBar.tsx`
- Create: `web/src/components/ResultView.tsx`

- [ ] **Step 1: Create UploadArea.tsx**

```tsx
import { useRef } from 'react'

interface UploadedFile {
  path: string
  name: string
  previewUrl?: string
}

interface Props {
  files: {
    audio?: UploadedFile
    lyrics?: UploadedFile
    cover?: UploadedFile
  }
  onFilesChange: (files: Props['files']) => void
}

export default function UploadArea({ files, onFilesChange }: Props) {
  const audioRef = useRef<HTMLInputElement>(null)
  const lyricsRef = useRef<HTMLInputElement>(null)
  const coverRef = useRef<HTMLInputElement>(null)

  async function handleUpload(
    file: File,
    key: 'audio' | 'lyrics' | 'cover',
  ) {
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch('/api/upload', { method: 'POST', body: formData })
    const data = await res.json()

    const uploaded: UploadedFile = { path: data.path, name: file.name }

    if (key === 'cover') {
      uploaded.previewUrl = URL.createObjectURL(file)
    }

    onFilesChange({ ...files, [key]: uploaded })
  }

  function renderZone(
    key: 'audio' | 'lyrics' | 'cover',
    label: string,
    accept: string,
    ref: React.RefObject<HTMLInputElement | null>,
  ) {
    const file = files[key]
    return (
      <div
        className={`upload-zone ${file ? 'uploaded' : ''}`}
        onClick={() => ref.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          const f = e.dataTransfer.files[0]
          if (f) handleUpload(f, key)
        }}
      >
        <input
          ref={ref}
          type="file"
          accept={accept}
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) handleUpload(f, key)
          }}
        />
        <p>{label}</p>
        {file && <p className="filename">{file.name}</p>}
        {key === 'cover' && file?.previewUrl && (
          <img src={file.previewUrl} alt="cover" className="cover-preview" />
        )}
      </div>
    )
  }

  return (
    <div className="section">
      <h2>Upload Files</h2>
      {renderZone('audio', 'Drop MP3 file here', '.mp3,audio/*', audioRef)}
      {renderZone('lyrics', 'Drop lyrics .txt file here', '.txt,text/*', lyricsRef)}
      {renderZone('cover', 'Drop cover image here', '.jpg,.jpeg,.png,image/*', coverRef)}
    </div>
  )
}
```

- [ ] **Step 2: Create ConfigPanel.tsx**

```tsx
interface Config {
  aspect: string
  theme: string
  lyricsStyle: string
}

interface Props {
  config: Config
  onConfigChange: (config: Config) => void
}

const THEMES = [
  { value: 'neon', label: 'Neon Pulse' },
  { value: 'vinyl', label: 'Vinyl Minimal' },
  { value: 'wave', label: 'Wave Groove' },
]

const LYRICS_STYLES = [
  { value: 'karaoke', label: 'KTV Highlight' },
  { value: 'fade', label: 'Fade In/Out' },
  { value: 'word-fill', label: 'Word Fill' },
]

export default function ConfigPanel({ config, onConfigChange }: Props) {
  function update(key: keyof Config, value: string) {
    onConfigChange({ ...config, [key]: value })
  }

  return (
    <div className="section">
      <h2>Settings</h2>

      <div className="radio-group">
        <h3 style={{ fontSize: '0.85rem', color: '#888', marginBottom: 8 }}>Aspect Ratio</h3>
        {['9:16', '16:9'].map((v) => (
          <label key={v}>
            <input
              type="radio"
              name="aspect"
              value={v}
              checked={config.aspect === v}
              onChange={() => update('aspect', v)}
            />
            {v}
          </label>
        ))}
      </div>

      <div className="radio-group">
        <h3 style={{ fontSize: '0.85rem', color: '#888', marginBottom: 8 }}>Theme</h3>
        {THEMES.map((t) => (
          <label key={t.value}>
            <input
              type="radio"
              name="theme"
              value={t.value}
              checked={config.theme === t.value}
              onChange={() => update('theme', t.value)}
            />
            {t.label}
          </label>
        ))}
      </div>

      <div className="radio-group">
        <h3 style={{ fontSize: '0.85rem', color: '#888', marginBottom: 8 }}>Lyrics Style</h3>
        {LYRICS_STYLES.map((s) => (
          <label key={s.value}>
            <input
              type="radio"
              name="lyricsStyle"
              value={s.value}
              checked={config.lyricsStyle === s.value}
              onChange={() => update('lyricsStyle', s.value)}
            />
            {s.label}
          </label>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create ProgressBar.tsx**

```tsx
interface Props {
  progress: number
  message: string
}

export default function ProgressBar({ progress, message }: Props) {
  return (
    <div className="section progress-section">
      <div className="progress-bar-bg">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.round(progress * 100)}%` }}
        />
      </div>
      <p className="progress-text">
        {Math.round(progress * 100)}% — {message}
      </p>
    </div>
  )
}
```

- [ ] **Step 4: Create ResultView.tsx**

```tsx
interface Props {
  videoUrl: string
  taskId: string
}

export default function ResultView({ videoUrl, taskId }: Props) {
  return (
    <div className="section result-section">
      <h2>Result</h2>
      <video src={videoUrl} controls />
      <div>
        <a className="download-btn" href={videoUrl} download={`mv_${taskId}.mp4`}>
          Download MP4
        </a>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Verify frontend builds**

Run:
```bash
cd web && npx tsc --noEmit && npx vite build
```
Expected: Build succeeds, `web/dist/` created

- [ ] **Step 6: Commit**

```bash
git add web/src/components/
git commit -m "feat: React frontend components — upload, config, progress, result"
```

---

### Task 14: Run Script and Final Integration

**Files:**
- Create: `run.py`
- Create: `.gitignore`

- [ ] **Step 1: Create run.py**

```python
#!/usr/bin/env python3
"""One-command launcher: starts FastAPI backend and serves the frontend."""
import subprocess
import sys
import os
from pathlib import Path


def main():
    # Build frontend if dist doesn't exist
    web_dir = Path("web")
    dist_dir = web_dir / "dist"

    if not dist_dir.exists():
        print("Building frontend...")
        if not (web_dir / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=str(web_dir), check=True)
        subprocess.run(["npx", "vite", "build"], cwd=str(web_dir), check=True)

    # Start FastAPI server
    print("Starting MV Generate server at http://localhost:8000")
    os.execvp(sys.executable, [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
venv/
uploads/
outputs/
web/node_modules/
web/dist/
*.egg-info/
.DS_Store
```

- [ ] **Step 3: Verify full server startup**

Run: `python run.py` (manual test — Ctrl+C to stop)
Expected: Server starts on port 8000, frontend is served

- [ ] **Step 4: Commit**

```bash
git add run.py .gitignore
git commit -m "feat: one-command launcher and gitignore"
```

---

### Task 15: End-to-End Smoke Test

- [ ] **Step 1: Prepare test files**

Create a short test MP3 (or use an existing one), a lyrics.txt, and a cover.jpg.

```bash
# Generate a 5-second test tone
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" -codec:a libmp3lame test_song.mp3

# Create test lyrics
cat > test_lyrics.txt << 'EOF'
Hello world
This is a test
Music video generator
Works great
EOF

# Create a test cover (solid color)
python -c "from PIL import Image; Image.new('RGB', (500, 500), (50, 100, 200)).save('test_cover.png')"
```

- [ ] **Step 2: Run CLI end-to-end**

Run:
```bash
python mvgenerate.py \
  --audio test_song.mp3 \
  --lyrics test_lyrics.txt \
  --cover test_cover.png \
  --theme neon \
  --lyrics-style karaoke \
  --title "Test Song" \
  --artist "Test Artist"
```
Expected: `output.mp4` is created, playable, shows cover -> disc -> lyrics

- [ ] **Step 3: Test with other themes and lyrics styles**

Run:
```bash
python mvgenerate.py --audio test_song.mp3 --lyrics test_lyrics.txt --cover test_cover.png --theme vinyl --lyrics-style fade --output vinyl_test.mp4
python mvgenerate.py --audio test_song.mp3 --lyrics test_lyrics.txt --cover test_cover.png --theme wave --lyrics-style word-fill --output wave_test.mp4
```
Expected: Both videos are created and playable

- [ ] **Step 4: Clean up test files, final commit**

```bash
rm -f test_song.mp3 test_lyrics.txt test_cover.png output.mp4 vinyl_test.mp4 wave_test.mp4
git add -A
git commit -m "chore: complete project structure"
```
