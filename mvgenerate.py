#!/usr/bin/env python3
# mvgenerate.py
"""MV Generate — CLI entry point."""
import argparse
import sys
from pathlib import Path

from config import (
    AspectRatio, Theme, LyricsStyle, GenerateConfig,
    RESOLUTIONS, FPS,
)
from align.lyrics_preprocessor import preprocess_lyrics_file
from align.lyrics_aligner import align_lyrics
from audio.analyzer import analyze_audio
from render.base import Renderer
from output.composer import compose_video_stream


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

    # Step 4: Render + stream to FFmpeg (no temp files)
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
    width, height = RESOLUTIONS[config.aspect]

    def frame_generator():
        for i in range(total_frames):
            time_s = i / FPS
            frame = renderer.render_frame(i, time_s, features, timed_lines)
            if i % FPS == 0:
                pct = 0.25 + 0.65 * (i / total_frames)
                report(f"Rendering frame {i}/{total_frames}...", pct)
            yield frame

    report("Rendering and encoding video...", 0.25)
    compose_video_stream(
        frame_generator(), config.audio_path, config.output_path,
        width, height, FPS,
    )

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
