# align/lyrics_aligner.py
from pathlib import Path
import mlx_whisper

from config import TimedLine, TimedWord

# MLX Whisper model repo (Apple Silicon optimized)
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def align_lyrics(audio_path: Path, lyrics_lines: list[str], model_repo: str = DEFAULT_MODEL) -> list[TimedLine]:
    """
    Align lyrics lines to audio using Whisper segment boundaries.

    Strategy: Use Whisper to detect vocal segments (when someone is singing),
    then map user lyrics lines to these segments in order. This avoids the
    unreliable character-matching approach — we only use Whisper for timing,
    not for text recognition.
    """
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        word_timestamps=True,
    )

    segments = result.get("segments", [])

    if not segments:
        duration = result.get("duration", 0.0)
        return _even_distribute(lyrics_lines, duration)

    total_duration = segments[-1].get("end", 0.0)

    # Collect segment boundaries (each segment ≈ one phrase of singing)
    seg_boundaries: list[tuple[float, float]] = []
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if end > start:
            seg_boundaries.append((start, end))

    if not seg_boundaries:
        return _even_distribute(lyrics_lines, total_duration)

    # Merge segments that are very close together (< 0.3s gap)
    merged: list[tuple[float, float]] = [seg_boundaries[0]]
    for start, end in seg_boundaries[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < 0.3:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Now map lyrics lines to segments
    num_lines = len(lyrics_lines)
    num_segs = len(merged)

    if num_lines == num_segs:
        # Perfect 1:1 mapping
        return _map_lines_to_segs(lyrics_lines, merged)

    if num_lines < num_segs:
        # More segments than lyrics lines — group segments
        return _distribute_fewer_lines(lyrics_lines, merged)

    # More lyrics lines than segments — split segments
    return _distribute_more_lines(lyrics_lines, merged, total_duration)


def _map_lines_to_segs(
    lines: list[str], segs: list[tuple[float, float]]
) -> list[TimedLine]:
    """1:1 mapping of lyrics lines to segments."""
    return [
        TimedLine(text=line, start=s, end=e)
        for line, (s, e) in zip(lines, segs)
    ]


def _distribute_fewer_lines(
    lines: list[str], segs: list[tuple[float, float]]
) -> list[TimedLine]:
    """More segments than lines: group consecutive segments per line."""
    num_lines = len(lines)
    num_segs = len(segs)

    # Distribute segments as evenly as possible across lines
    timed: list[TimedLine] = []
    seg_idx = 0
    for i, line in enumerate(lines):
        # How many segments for this line
        remaining_lines = num_lines - i
        remaining_segs = num_segs - seg_idx
        count = max(1, remaining_segs // remaining_lines)

        group_start = segs[seg_idx][0]
        group_end = segs[min(seg_idx + count - 1, num_segs - 1)][1]
        timed.append(TimedLine(text=line, start=group_start, end=group_end))
        seg_idx += count

    return timed


def _distribute_more_lines(
    lines: list[str], segs: list[tuple[float, float]], total_duration: float
) -> list[TimedLine]:
    """More lines than segments: subdivide segments to fit all lines."""
    num_lines = len(lines)
    num_segs = len(segs)

    # Calculate total singing time
    total_sing_time = sum(e - s for s, e in segs)

    timed: list[TimedLine] = []
    line_idx = 0

    for seg_start, seg_end in segs:
        seg_duration = seg_end - seg_start
        # How many lines go into this segment (proportional to duration)
        remaining_lines = num_lines - line_idx
        remaining_segs_time = sum(e - s for s, e in segs[segs.index((seg_start, seg_end)):])
        if remaining_segs_time > 0:
            count = max(1, round(remaining_lines * seg_duration / remaining_segs_time))
        else:
            count = remaining_lines
        count = min(count, remaining_lines)

        # Subdivide this segment equally
        sub_duration = seg_duration / count
        for j in range(count):
            sub_start = seg_start + j * sub_duration
            sub_end = seg_start + (j + 1) * sub_duration
            if line_idx < num_lines:
                timed.append(TimedLine(text=lines[line_idx], start=sub_start, end=sub_end))
                line_idx += 1

    # Any remaining lines get distributed after last segment
    if line_idx < num_lines:
        remaining = lines[line_idx:]
        last_end = timed[-1].end if timed else 0
        gap = total_duration - last_end
        per_line = max(0.5, gap / len(remaining)) if remaining else 0
        for i, line in enumerate(remaining):
            timed.append(TimedLine(
                text=line,
                start=last_end + i * per_line,
                end=last_end + (i + 1) * per_line,
            ))

    return timed


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
