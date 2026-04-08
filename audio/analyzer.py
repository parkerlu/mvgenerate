import numpy as np
import librosa
from pathlib import Path

from config import AudioFeatures, FPS

NUM_BANDS = 16


def analyze_audio(audio_path: Path) -> AudioFeatures:
    """Extract per-frame audio features for visualization."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    num_frames = int(duration * FPS)

    hop_length = sr // FPS

    # RMS energy per frame
    rms_raw = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    rms_resampled = np.interp(
        np.linspace(0, len(rms_raw) - 1, num_frames),
        np.arange(len(rms_raw)),
        rms_raw,
    )
    rms_max = rms_resampled.max() if rms_resampled.max() > 0 else 1.0
    rms = (rms_resampled / rms_max).tolist()

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NUM_BANDS, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_range = mel_max - mel_min if mel_max > mel_min else 1.0
    mel_norm = (mel_db - mel_min) / mel_range

    spectrum: list[list[float]] = []
    for i in range(num_frames):
        src_idx = int(i * mel_norm.shape[1] / num_frames)
        src_idx = min(src_idx, mel_norm.shape[1] - 1)
        spectrum.append(mel_norm[:, src_idx].tolist())

    # Beat detection
    tempo, beat_frame_indices = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frame_indices, sr=sr, hop_length=hop_length)
    beat_frames = [int(t * FPS) for t in beat_times if int(t * FPS) < num_frames]

    return AudioFeatures(
        rms=rms,
        spectrum=spectrum,
        beat_frames=beat_frames,
        duration=duration,
    )


def detect_chorus(audio_path: Path, min_duration: float = 15.0) -> tuple[float, float]:
    """
    Detect the chorus (highest energy section) of a song.

    Returns (start_time, end_time) in seconds.
    Uses a sliding window over RMS energy to find the most energetic section.
    """
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Sliding window: find the window with highest average energy
    window_frames = int(min_duration * sr / hop_length)
    if window_frames >= len(rms):
        return (0.0, duration)

    best_start = 0
    best_energy = 0.0

    # Compute cumulative sum for efficient sliding window
    cum_sum = np.cumsum(rms)
    for start in range(len(rms) - window_frames):
        end = start + window_frames
        energy = cum_sum[end] - cum_sum[start]
        if energy > best_energy:
            best_energy = energy
            best_start = start

    start_time = librosa.frames_to_time(best_start, sr=sr, hop_length=hop_length)
    end_time = librosa.frames_to_time(best_start + window_frames, sr=sr, hop_length=hop_length)

    # Snap to nearest beat for cleaner cuts
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    if len(beat_times) > 0:
        # Snap start to nearest earlier beat
        earlier_beats = beat_times[beat_times <= start_time]
        if len(earlier_beats) > 0:
            start_time = earlier_beats[-1]
        # Snap end to nearest later beat
        later_beats = beat_times[beat_times >= end_time]
        if len(later_beats) > 0:
            end_time = later_beats[0]

    return (round(start_time, 2), round(end_time, 2))
