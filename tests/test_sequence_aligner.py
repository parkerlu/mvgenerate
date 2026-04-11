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
