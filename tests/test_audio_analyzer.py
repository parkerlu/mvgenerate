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
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, sr)
    return path


def test_analyze_returns_audio_features(sine_wav):
    features = analyze_audio(sine_wav)
    expected_frames = int(features.duration * FPS)
    assert abs(len(features.rms) - expected_frames) <= 1
    assert abs(len(features.spectrum) - expected_frames) <= 1
    assert features.duration == pytest.approx(2.0, abs=0.1)


def test_rms_values_normalized(sine_wav):
    features = analyze_audio(sine_wav)
    assert all(0.0 <= v <= 1.0 for v in features.rms)


def test_spectrum_has_bands(sine_wav):
    features = analyze_audio(sine_wav)
    assert len(features.spectrum[0]) >= 8


def test_beat_frames_are_valid_indices(sine_wav):
    features = analyze_audio(sine_wav)
    num_frames = len(features.rms)
    for bf in features.beat_frames:
        assert 0 <= bf < num_frames
