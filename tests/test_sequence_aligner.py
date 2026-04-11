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
