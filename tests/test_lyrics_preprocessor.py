from align.lyrics_preprocessor import preprocess_lyrics


def test_removes_section_markers():
    raw = "[Verse]\nfirst line\n[Chorus]\nsecond line"
    result = preprocess_lyrics(raw)
    assert result == ["first line", "second line"]


def test_removes_empty_lines():
    raw = "line one\n\n\nline two\n\n"
    result = preprocess_lyrics(raw)
    assert result == ["line one", "line two"]


def test_trims_whitespace():
    raw = "  hello world  \n  foo bar  "
    result = preprocess_lyrics(raw)
    assert result == ["hello world", "foo bar"]


def test_removes_various_suno_markers():
    raw = "[Intro]\n[Verse 1]\nlyric\n[Pre-Chorus]\n[Chorus]\nchorus line\n[Bridge]\n[Outro]"
    result = preprocess_lyrics(raw)
    assert result == ["lyric", "chorus line"]


def test_handles_brackets_in_lyrics():
    raw = "she said [softly] hello"
    result = preprocess_lyrics(raw)
    # Lines that are ONLY a bracket marker get removed; brackets embedded in lyrics stay
    assert result == ["she said [softly] hello"]


def test_empty_input():
    assert preprocess_lyrics("") == []
    assert preprocess_lyrics("\n\n\n") == []


def test_reads_from_file(tmp_path):
    from align.lyrics_preprocessor import preprocess_lyrics_file

    f = tmp_path / "lyrics.txt"
    f.write_text("[Verse]\nhello\n\nworld\n", encoding="utf-8")
    result = preprocess_lyrics_file(f)
    assert result == ["hello", "world"]
