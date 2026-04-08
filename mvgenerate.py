#!/usr/bin/env python3
# mvgenerate.py
"""MV Generate — CLI entry point."""
import argparse
import sys
from pathlib import Path

from config import (
    AspectRatio, Theme, LyricsStyle, GenerateConfig, GenerateMode,
    RESOLUTIONS, FPS, COVER_DURATION, TRANSITION_DURATION, OUTRO_DURATION,
)
from align.lyrics_preprocessor import preprocess_lyrics_file
from align.lyrics_aligner import align_lyrics
from align.cache import get_cached, save_cache
from audio.analyzer import analyze_audio, detect_chorus
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

    # Step 1.5: Detect chorus if needed
    chorus_start, chorus_end = None, None
    if config.mode == GenerateMode.CHORUS:
        report("Detecting chorus section...", 0.02)
        chorus_start, chorus_end = detect_chorus(config.audio_path)
        report(f"Chorus detected: {chorus_start:.1f}s - {chorus_end:.1f}s", 0.04)

    # Step 2: Check cache first, then align
    import threading
    import time

    cached = get_cached(config.audio_path, lyrics_lines)
    if cached:
        report("Using cached lyrics alignment", 0.19)
        timed_lines = cached
    else:
        report("Aligning lyrics with audio...", 0.05)
        align_result: list = []
        align_error: list = []
        align_done = threading.Event()

        def _align():
            try:
                result = align_lyrics(config.audio_path, lyrics_lines)
                align_result.append(result)
            except Exception as e:
                align_error.append(e)
            finally:
                align_done.set()

        thread = threading.Thread(target=_align, daemon=True)
        thread.start()

        start_time = time.time()
        while not align_done.is_set():
            elapsed = int(time.time() - start_time)
            report(f"Aligning lyrics... ({elapsed}s elapsed)", 0.05 + min(0.14, elapsed * 0.001))
            align_done.wait(timeout=2.0)

        if align_error:
            raise align_error[0]
        timed_lines = align_result[0]
        elapsed = int(time.time() - start_time)
        report(f"Lyrics aligned ({elapsed}s)", 0.19)

        # Save to cache for next time
        save_cache(config.audio_path, lyrics_lines, timed_lines)

    # Filter lyrics to chorus section if needed
    if chorus_start is not None:
        timed_lines = [
            line for line in timed_lines
            if line.end > chorus_start and line.start < chorus_end
        ]
        # Shift timestamps relative to chorus start
        for line in timed_lines:
            line.start = max(0, line.start - chorus_start)
            line.end = line.end - chorus_start
            for word in line.words:
                word.start = max(0, word.start - chorus_start)
                word.end = word.end - chorus_start

    # Step 3: Analyze audio
    report("Analyzing audio...", 0.20)
    features = analyze_audio(config.audio_path)

    # Trim audio features to chorus if needed
    if chorus_start is not None:
        start_frame = int(chorus_start * FPS)
        end_frame = int(chorus_end * FPS)
        features.rms = features.rms[start_frame:end_frame]
        features.spectrum = features.spectrum[start_frame:end_frame]
        features.beat_frames = [
            f - start_frame for f in features.beat_frames
            if start_frame <= f < end_frame
        ]
        features.duration = chorus_end - chorus_start

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

    # Determine audio source (full or trimmed)
    audio_for_video = config.audio_path
    tmp_chorus_audio = None
    if chorus_start is not None:
        import subprocess, tempfile
        tmp_chorus_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(config.audio_path),
            "-ss", str(chorus_start),
            "-to", str(chorus_end),
            "-c", "copy",
            tmp_chorus_audio.name,
        ], capture_output=True)
        audio_for_video = tmp_chorus_audio.name

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
        frame_generator(), audio_for_video, config.output_path,
        width, height, FPS,
    )

    # Clean up temp chorus audio
    if tmp_chorus_audio is not None:
        import os
        os.unlink(tmp_chorus_audio.name)

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
    parser.add_argument("--mode", choices=["full", "chorus"], default="full", help="Generate full song or auto-detect chorus (default: full)")
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
        mode=GenerateMode(args.mode),
        title=args.title,
        artist=args.artist,
    )

    generate(config)


if __name__ == "__main__":
    main()
