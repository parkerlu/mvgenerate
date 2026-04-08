import subprocess
from pathlib import Path
from PIL import Image

from output.composer import compose_video


def test_compose_creates_mp4(tmp_path):
    """Test that composer creates a valid MP4 from frames + audio."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(30):
        img = Image.new("RGB", (320, 240), (i * 8, 100, 200))
        img.save(frames_dir / f"frame_{i:06d}.png")

    audio_path = tmp_path / "silence.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", "1", str(audio_path),
    ], capture_output=True)

    output_path = tmp_path / "output.mp4"
    compose_video(frames_dir, audio_path, output_path, fps=30)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
