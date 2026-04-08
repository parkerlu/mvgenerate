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
