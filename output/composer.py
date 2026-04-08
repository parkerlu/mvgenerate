import subprocess
from pathlib import Path
from typing import Iterator
from PIL import Image


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


def compose_video_stream(
    frame_iter: Iterator[Image.Image],
    audio_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int = 30,
) -> None:
    """Stream frames directly to FFmpeg via pipe (no disk I/O)."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for frame in frame_iter:
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"FFmpeg failed:\n{stderr}")
