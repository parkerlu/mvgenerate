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
