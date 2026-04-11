"""End-to-end smoke test on the sample song.

This test is SLOW (runs real Whisper + real LLM) and requires:
- sample/这温暖的家.mp3
- sample/这温暖的家.txt
- A valid DEEPSEEK_API_KEY in .env

Run with: pytest tests/test_alignment_e2e.py -v -s
Skip in CI by default.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from align.lyrics_aligner import align_lyrics
from align.lyrics_preprocessor import preprocess_lyrics_file

load_dotenv()

SAMPLE_DIR = Path(__file__).parent.parent / "sample"
AUDIO = SAMPLE_DIR / "这温暖的家.mp3"
LYRICS = SAMPLE_DIR / "这温暖的家.txt"


@pytest.mark.skipif(
    not AUDIO.exists() or not LYRICS.exists(),
    reason="sample audio/lyrics missing",
)
def test_e2e_alignment_produces_monotonic_timeline():
    lyric_lines = preprocess_lyrics_file(LYRICS)
    assert len(lyric_lines) > 10  # sanity: 这温暖的家 有很多行

    timed = align_lyrics(AUDIO, lyric_lines)

    # 1. 每行都有输出
    assert len(timed) == len(lyric_lines)

    # 2. 时间戳单调不减（后一行 start >= 前一行 start）
    for i in range(1, len(timed)):
        assert timed[i].start >= timed[i - 1].start, (
            f"Line {i} starts before line {i-1}: "
            f"{timed[i-1].start:.2f} -> {timed[i].start:.2f}"
        )

    # 3. 每行 end >= start
    for i, t in enumerate(timed):
        assert t.end >= t.start, f"Line {i} has end < start"

    # 4. 打印结果供人工检查
    print("\n=== Alignment result ===")
    for t in timed:
        print(f"  [{t.start:6.2f} -> {t.end:6.2f}]  {t.text}")
