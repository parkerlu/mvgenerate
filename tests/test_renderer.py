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
        beat_frames=[i for i in range(0, num_frames, FPS)],
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
