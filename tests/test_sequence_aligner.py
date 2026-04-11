from align.sequence_aligner import sim


def test_sim_identical_chinese():
    assert sim("你好世界", "你好世界") == 1.0


def test_sim_completely_different_chinese():
    assert sim("你好世界", "再见月亮") == 0.0


def test_sim_ignores_punctuation():
    # Punctuation differences should not affect similarity
    assert sim("你好，世界！", "你好世界") == 1.0


def test_sim_ignores_whitespace():
    assert sim("你好 世界", "你好世界") == 1.0


def test_sim_partial_match_chinese():
    # 3 of 4 chars match; should be in (0.4, 0.7) range
    result = sim("你好世界", "你好宇宙")
    assert 0.4 < result < 0.7


def test_sim_empty_strings():
    assert sim("", "") == 1.0
    assert sim("", "你好") == 0.0


from align.sequence_aligner import Segment, skip_cost


def test_skip_cost_pure_filler():
    # Pure filler segments: almost free to skip
    assert skip_cost(Segment(text="嗯", start=0, end=0.5)) == -0.05
    assert skip_cost(Segment(text="啊啊", start=0, end=0.5)) == -0.05
    assert skip_cost(Segment(text="哦 哦", start=0, end=0.5)) == -0.05  # whitespace normalized away


def test_skip_cost_empty_text():
    # Empty or whitespace-only segments should return the filler penalty
    assert skip_cost(Segment(text="", start=0, end=0.5)) == -0.05
    assert skip_cost(Segment(text="   ", start=0, end=0.5)) == -0.05


def test_skip_cost_very_short():
    # Non-filler but very short (<=2 chars)
    assert skip_cost(Segment(text="好的", start=0, end=0.5)) == -0.10


def test_skip_cost_normal_length():
    # Normal length: -0.05 * len
    assert skip_cost(Segment(text="你好世界吗", start=0, end=2.0)) == -0.25  # 5 chars


def test_skip_cost_capped():
    # Long segment capped at -0.5
    long_text = "一二三四五六七八九十十一十二"
    assert skip_cost(Segment(text=long_text, start=0, end=5.0)) == -0.5


from align.sequence_aligner import align, LineAlignment


def test_align_perfect_one_to_one():
    lyrics = ["你好世界", "再见月亮"]
    segments = [
        Segment(text="你好世界", start=1.0, end=3.0),
        Segment(text="再见月亮", start=4.0, end=6.0),
    ]
    alignments, confidences = align(lyrics, segments)

    assert len(alignments) == 2
    assert alignments[0] == LineAlignment(line_idx=0, segment_idxs=[0])
    assert alignments[1] == LineAlignment(line_idx=1, segment_idxs=[1])
    assert confidences[0] == 1.0
    assert confidences[1] == 1.0


def test_align_one_lyric_two_segments():
    # Whisper split a lyric line into two segments
    lyrics = ["你曾说过海枯石烂"]
    segments = [
        Segment(text="你曾说过", start=1.0, end=2.5),
        Segment(text="海枯石烂", start=2.5, end=4.0),
    ]
    alignments, confidences = align(lyrics, segments)

    assert len(alignments) == 1
    assert alignments[0].segment_idxs == [0, 1]
    assert confidences[0] > 0.9


def test_align_two_lyrics_one_segment():
    # Whisper merged two lyric lines into one segment
    lyrics = ["你好世界", "再见月亮"]
    segments = [
        Segment(text="你好世界再见月亮", start=1.0, end=5.0),
    ]
    alignments, confidences = align(lyrics, segments)

    assert len(alignments) == 2
    assert alignments[0].segment_idxs == [0]
    assert alignments[1].segment_idxs == [0]


def test_align_skips_filler_segment():
    # Whisper 识别出了衬词 "嗯"，不在歌词里
    lyrics = ["你曾说过海枯石烂", "如今却各自天涯"]
    segments = [
        Segment(text="你曾说过海枯石烂", start=1.0, end=4.0),
        Segment(text="嗯", start=4.0, end=4.3),  # 衬词
        Segment(text="如今却各自天涯", start=4.3, end=7.0),
    ]
    alignments, confidences = align(lyrics, segments)

    assert alignments[0].segment_idxs == [0]
    assert alignments[1].segment_idxs == [2]
    assert all(c > 0.9 for c in confidences)


def test_align_whisper_dropped_a_char():
    # Whisper 少听了一个字，相似度应该仍然高
    lyrics = ["你曾说过海枯石烂"]
    segments = [
        Segment(text="你曾说海枯石烂", start=1.0, end=4.0),  # 漏了"过"
    ]
    alignments, confidences = align(lyrics, segments)

    assert alignments[0].segment_idxs == [0]
    assert confidences[0] > 0.85  # 高但不是完美


def test_align_completely_unmatched_lyric_gets_low_confidence():
    # 歌词里有一行 Whisper 完全没识别到
    lyrics = ["你好世界", "这行没人唱"]
    segments = [
        Segment(text="你好世界", start=1.0, end=3.0),
    ]
    alignments, confidences = align(lyrics, segments)

    assert alignments[0].segment_idxs == [0]
    assert confidences[0] == 1.0
    # 第二行也被分配到相同segment，但置信度为 0（相似度为0）
    assert alignments[1].segment_idxs == [0]
    assert confidences[1] == 0.0
