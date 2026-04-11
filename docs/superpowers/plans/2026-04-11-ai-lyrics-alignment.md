# AI 歌词对齐实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `align/lyrics_aligner.py` 从"按字符比例切时间戳"重写为"基于序列比对 + LLM 精修"的两阶段对齐，解决 70% 歌词不对齐问题。

**Architecture:** 经典 DP 序列对齐（中文字符级相似度 + 衬词感知 skip）为主力，低置信度窗口送 DeepSeek/Claude 做语义精修。LLM 只返回索引映射，时间戳永远从 Whisper 片段查，保证 LLM 不会污染时间精度。

**Tech Stack:** Python 3.11+, `mlx_whisper`（已有）, `difflib`（标准库）, `openai` SDK（调 DeepSeek，OpenAI 兼容）, `anthropic` SDK（可选 Claude）, `python-dotenv`, `pytest`。

**Spec:** `docs/superpowers/specs/2026-04-11-ai-lyrics-alignment-design.md`

**Discovery during planning:** `align/lyrics_preprocessor.py` 已经在调用 `align_lyrics()` 之前移除 `[Chorus]` 等结构标记和空行。所以新代码拿到的 `lyrics_lines` 已是纯歌词，不需要重做预处理。Spec 第 "组件详细设计 · 1. 歌词预处理" 那一节在实现里略过。

---

## 文件结构

**新增**：
- `align/sequence_aligner.py` — 纯函数：DP 对齐算法
- `align/llm_fallback.py` — LLM provider 抽象 + 精修逻辑
- `tests/test_sequence_aligner.py` — 算法单元测试
- `tests/test_llm_fallback.py` — fallback 单元测试（mock provider）
- `tests/test_alignment_e2e.py` — 端到端 smoke test（需真实音频）
- `.env.example` — API key 占位

**修改**：
- `align/lyrics_aligner.py` — 重写 `align_lyrics()` 为编排层
- `requirements.txt` — 新增 `openai`、`anthropic`、`python-dotenv`

**不动**：
- `align/transcriber.py`
- `align/lyrics_preprocessor.py`
- `align/cache.py`
- `mvgenerate.py`
- `server/routes.py`

---

## Task 1: 新增依赖与环境配置

**Files:**
- Modify: `requirements.txt`
- Create: `.env.example`

- [ ] **Step 1: 追加依赖到 `requirements.txt`**

在文件末尾追加：

```
openai==1.54.0
anthropic==0.40.0
python-dotenv==1.0.1
```

- [ ] **Step 2: 安装新依赖**

Run: `pip install -r requirements.txt`
Expected: 三个新包成功安装，无版本冲突。

- [ ] **Step 3: 创建 `.env.example`**

```
# LLM provider for lyrics alignment fallback
# Options: deepseek | claude
LYRICS_ALIGNER_LLM_PROVIDER=deepseek

# Set to false to disable LLM fallback entirely (classical alignment only)
LYRICS_ALIGNER_LLM_ENABLED=true

# DeepSeek API key (https://platform.deepseek.com)
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Anthropic API key (https://console.anthropic.com) — only needed if provider=claude
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

- [ ] **Step 4: 确认本地 `.env` 文件存在且已填 `DEEPSEEK_API_KEY`**

Run: `test -f .env && grep -q DEEPSEEK_API_KEY .env && echo OK || echo MISSING`
Expected: `OK`。如果 `MISSING`，把 `.env.example` 复制为 `.env` 并填入真实 key。

- [ ] **Step 5: 确认 `.env` 在 `.gitignore` 里**

Run: `grep -q '^\.env$' .gitignore && echo OK || echo MISSING`
Expected: `OK`。如果 `MISSING`，追加 `.env` 到 `.gitignore`。

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "chore: add deepseek/claude/dotenv deps for lyrics alignment"
```

---

## Task 2: 定义 Segment 类型与相似度函数

**Files:**
- Create: `align/sequence_aligner.py`
- Create: `tests/test_sequence_aligner.py`

- [ ] **Step 1: 写失败的测试 —— 相似度函数基础行为**

Create `tests/test_sequence_aligner.py` with:

```python
from align.sequence_aligner import sim


def test_sim_identical_chinese():
    assert sim("你好世界", "你好世界") == 1.0


def test_sim_completely_different_chinese():
    assert sim("你好世界", "再见月亮") == 0.0


def test_sim_ignores_punctuation():
    # 标点差异不应影响相似度
    assert sim("你好，世界！", "你好世界") == 1.0


def test_sim_ignores_whitespace():
    assert sim("你好 世界", "你好世界") == 1.0


def test_sim_partial_match_chinese():
    # 4 个字中 3 个相同，应该在 0.7 附近
    result = sim("你好世界", "你好宇宙")
    assert 0.4 < result < 0.7


def test_sim_empty_strings():
    assert sim("", "") == 1.0
    assert sim("", "你好") == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_sequence_aligner.py -v`
Expected: `ImportError: cannot import name 'sim' from 'align.sequence_aligner'`

- [ ] **Step 3: 实现 `sequence_aligner.py` 骨架与 `sim`**

Create `align/sequence_aligner.py`:

```python
"""Classical sequence alignment between user lyrics and Whisper segments."""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class Segment:
    """A Whisper transcription segment with timing."""
    text: str
    start: float
    end: float


def _normalize(s: str) -> str:
    """Keep only CJK chars and alphanumerics; drop punctuation and whitespace."""
    return ''.join(
        c for c in s
        if c.isalnum() or '\u4e00' <= c <= '\u9fff'
    )


def sim(a: str, b: str) -> float:
    """Character-level similarity in [0.0, 1.0]. Punctuation/whitespace insensitive."""
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_sequence_aligner.py -v`
Expected: 6 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add align/sequence_aligner.py tests/test_sequence_aligner.py
git commit -m "feat(align): add Segment dataclass and sim() similarity function"
```

---

## Task 3: 实现 skip_cost 函数

**Files:**
- Modify: `align/sequence_aligner.py`
- Modify: `tests/test_sequence_aligner.py`

- [ ] **Step 1: 追加失败测试**

Append to `tests/test_sequence_aligner.py`:

```python
from align.sequence_aligner import Segment, skip_cost


def test_skip_cost_pure_filler():
    # 纯衬词片段：几乎免费跳过
    assert skip_cost(Segment(text="嗯", start=0, end=0.5)) == -0.05
    assert skip_cost(Segment(text="啊啊", start=0, end=0.5)) == -0.05
    assert skip_cost(Segment(text="哦 哦", start=0, end=0.5)) == -0.05  # 空格会被 normalize 掉


def test_skip_cost_very_short():
    # 非衬词但极短（≤2 字）
    assert skip_cost(Segment(text="好的", start=0, end=0.5)) == -0.10


def test_skip_cost_normal_length():
    # 普通长度：0.05 * len
    assert skip_cost(Segment(text="你好世界吗", start=0, end=2.0)) == -0.25  # 5 字


def test_skip_cost_capped():
    # 长片段被上限 0.5 兜住
    long_text = "一二三四五六七八九十十一十二"
    assert skip_cost(Segment(text=long_text, start=0, end=5.0)) == -0.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_sequence_aligner.py::test_skip_cost_pure_filler -v`
Expected: `ImportError: cannot import name 'skip_cost'`

- [ ] **Step 3: 实现 `skip_cost`**

Append to `align/sequence_aligner.py`:

```python
_FILLERS = set("嗯啊哦呀呃喔唉嘿哟噢咦哈")


def skip_cost(segment: Segment) -> float:
    """Cost of skipping a Whisper segment in DP alignment.

    Fillers and short segments cost almost nothing to skip; long segments
    cost more, so the DP strongly prefers matching them to some lyric line.
    Returns a negative number (penalty).
    """
    text = _normalize(segment.text)
    if not text:
        return -0.05
    if all(c in _FILLERS for c in text):
        return -0.05
    if len(text) <= 2:
        return -0.10
    return -min(0.5, 0.05 * len(text))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_sequence_aligner.py -v`
Expected: 全部 10 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add align/sequence_aligner.py tests/test_sequence_aligner.py
git commit -m "feat(align): add skip_cost for filler-aware segment skipping"
```

---

## Task 4: DP 对齐核心 —— 简单 1:1 场景

**Files:**
- Modify: `align/sequence_aligner.py`
- Modify: `tests/test_sequence_aligner.py`

- [ ] **Step 1: 写失败的测试 —— 1:1 完美匹配**

Append to `tests/test_sequence_aligner.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_sequence_aligner.py::test_align_perfect_one_to_one -v`
Expected: `ImportError: cannot import name 'align'`

- [ ] **Step 3: 实现 `LineAlignment` 与 `align` 的骨架（仅 1:1）**

Append to `align/sequence_aligner.py`:

```python
NEG_INF = float('-inf')


@dataclass
class LineAlignment:
    line_idx: int
    segment_idxs: list[int]


def align(
    lyric_lines: list[str],
    segments: list[Segment],
) -> tuple[list[LineAlignment], list[float]]:
    """Align lyric lines to Whisper segments using DP.

    Returns (alignments, confidences). alignments[i] maps lyric_lines[i]
    to a list of segment indices (possibly empty if no match).
    confidences[i] is in [0.0, 1.0].
    """
    n = len(lyric_lines)
    m = len(segments)

    if n == 0:
        return [], []
    if m == 0:
        return [LineAlignment(i, []) for i in range(n)], [0.0] * n

    # dp[i][j] = best score after processing first i lyric lines and first j segments
    # parent[i][j] = (prev_i, prev_j, action) for backtracking
    dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == NEG_INF:
                continue

            # Action: match lyric i to segment j (1:1)
            if i < n and j < m:
                score = dp[i][j] + sim(lyric_lines[i], segments[j].text)
                if score > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = score
                    parent[i + 1][j + 1] = (i, j, 'match_1_1')

            # Action: skip lyric line (gap / no match)
            if i < n:
                score = dp[i][j] - 0.5
                if score > dp[i + 1][j]:
                    dp[i + 1][j] = score
                    parent[i + 1][j] = (i, j, 'skip_lyric')

            # Action: skip segment (filler / backing vocal)
            if j < m:
                score = dp[i][j] + skip_cost(segments[j])
                if score > dp[i][j + 1]:
                    dp[i][j + 1] = score
                    parent[i][j + 1] = (i, j, 'skip_segment')

    # Backtrack from (n, m)
    alignments: list[LineAlignment] = [LineAlignment(i, []) for i in range(n)]
    i, j = n, m
    while (i, j) != (0, 0):
        p = parent[i][j]
        if p is None:
            break
        prev_i, prev_j, action = p
        if action == 'match_1_1':
            alignments[prev_i].segment_idxs.append(prev_j)
        i, j = prev_i, prev_j

    # Compute confidences
    confidences: list[float] = []
    for a in alignments:
        if not a.segment_idxs:
            confidences.append(0.0)
        else:
            matched_text = ''.join(segments[k].text for k in a.segment_idxs)
            confidences.append(sim(lyric_lines[a.line_idx], matched_text))

    return alignments, confidences
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_sequence_aligner.py::test_align_perfect_one_to_one -v`
Expected: PASS。

- [ ] **Step 5: 运行全部测试确认没回归**

Run: `pytest tests/test_sequence_aligner.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add align/sequence_aligner.py tests/test_sequence_aligner.py
git commit -m "feat(align): DP sequence alignment with 1:1 matching and skip actions"
```

---

## Task 5: DP 扩展 —— 1:2 和 2:1 合并

**Files:**
- Modify: `align/sequence_aligner.py`
- Modify: `tests/test_sequence_aligner.py`

- [ ] **Step 1: 写失败的测试 —— Whisper 把一行切成两段**

Append to `tests/test_sequence_aligner.py`:

```python
def test_align_one_lyric_two_segments():
    # Whisper 把一行歌词切成了两个片段
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
    # Whisper 把两行歌词合成了一个片段
    lyrics = ["你好世界", "再见月亮"]
    segments = [
        Segment(text="你好世界再见月亮", start=1.0, end=5.0),
    ]
    alignments, confidences = align(lyrics, segments)

    assert len(alignments) == 2
    assert alignments[0].segment_idxs == [0]
    assert alignments[1].segment_idxs == [0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_sequence_aligner.py::test_align_one_lyric_two_segments tests/test_sequence_aligner.py::test_align_two_lyrics_one_segment -v`
Expected: 两个都 FAIL（合并动作还没实现）。

- [ ] **Step 3: 在 DP 循环里追加两个合并动作**

In `align/sequence_aligner.py`, inside the `for i in range(n + 1): for j in range(m + 1):` loop, after the existing three actions, add:

```python
            # Action: match lyric i to segments j, j+1 (1:2)
            if i < n and j + 1 < m:
                merged = segments[j].text + segments[j + 1].text
                score = dp[i][j] + sim(lyric_lines[i], merged)
                if score > dp[i + 1][j + 2]:
                    dp[i + 1][j + 2] = score
                    parent[i + 1][j + 2] = (i, j, 'match_1_2')

            # Action: match lyrics i, i+1 to segment j (2:1)
            if i + 1 < n and j < m:
                merged = lyric_lines[i] + lyric_lines[i + 1]
                score = dp[i][j] + sim(merged, segments[j].text)
                if score > dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = score
                    parent[i + 2][j + 1] = (i, j, 'match_2_1')
```

And update the backtrack block to handle the new actions:

```python
    while (i, j) != (0, 0):
        p = parent[i][j]
        if p is None:
            break
        prev_i, prev_j, action = p
        if action == 'match_1_1':
            alignments[prev_i].segment_idxs.append(prev_j)
        elif action == 'match_1_2':
            alignments[prev_i].segment_idxs.extend([prev_j, prev_j + 1])
        elif action == 'match_2_1':
            alignments[prev_i].segment_idxs.append(prev_j)
            alignments[prev_i + 1].segment_idxs.append(prev_j)
        i, j = prev_i, prev_j
```

- [ ] **Step 4: 运行新测试确认通过**

Run: `pytest tests/test_sequence_aligner.py::test_align_one_lyric_two_segments tests/test_sequence_aligner.py::test_align_two_lyrics_one_segment -v`
Expected: PASS。

- [ ] **Step 5: 运行全部测试确认没回归**

Run: `pytest tests/test_sequence_aligner.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add align/sequence_aligner.py tests/test_sequence_aligner.py
git commit -m "feat(align): support 1:2 and 2:1 lyric-segment merging in DP"
```

---

## Task 6: DP 鲁棒性 —— 衬词与 Whisper 漏字

**Files:**
- Modify: `tests/test_sequence_aligner.py`

- [ ] **Step 1: 写测试覆盖真实场景**

Append to `tests/test_sequence_aligner.py`:

```python
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
    # 第二行没匹配，置信度应为 0
    assert alignments[1].segment_idxs == []
    assert confidences[1] == 0.0
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_sequence_aligner.py -v`
Expected: 全部 PASS（算法应该已经能处理这些场景，这是验证性测试）。

- [ ] **Step 3: 如果有失败，调整 skip_cost 或添加诊断 print 排查**

如果 `test_align_skips_filler_segment` 失败，很可能是 DP 优先匹配了衬词段。检查 skip_cost 返回值，确认 `-0.05` 比 `sim(lyrics, 嗯)` 得分高。

- [ ] **Step 4: Commit**

```bash
git add tests/test_sequence_aligner.py
git commit -m "test(align): cover filler skipping and Whisper inaccuracy cases"
```

---

## Task 7: LLM Provider 抽象

**Files:**
- Create: `align/llm_fallback.py`
- Create: `tests/test_llm_fallback.py`

- [ ] **Step 1: 写失败测试 —— Provider 接口与 DeepSeek 客户端初始化**

Create `tests/test_llm_fallback.py`:

```python
import os
from unittest.mock import MagicMock, patch

import pytest

from align.llm_fallback import DeepSeekProvider, ClaudeProvider, LLMProvider


def test_deepseek_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    p = DeepSeekProvider()
    assert p.api_key == "sk-test-deepseek"


def test_deepseek_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekProvider()


def test_claude_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = ClaudeProvider()
    assert p.api_key == "sk-ant-test"


def test_deepseek_chat_calls_openai_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    p = DeepSeekProvider()

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"ok":true}'))]
    with patch.object(p._client.chat.completions, "create", return_value=fake_response) as mock_create:
        result = p.chat(system="sys", user="usr")

    assert result == '{"ok":true}'
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-chat"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"] == "sys"
    assert call_kwargs["messages"][1]["role"] == "user"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: `ImportError: cannot import name 'DeepSeekProvider'`

- [ ] **Step 3: 实现 Provider 抽象与实现类**

Create `align/llm_fallback.py`:

```python
"""LLM fallback for refining low-confidence lyric alignment windows."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """Send a system+user message and return the assistant's text response."""


class DeepSeekProvider(LLMProvider):
    """DeepSeek via OpenAI-compatible API."""

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in environment")
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

    def chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class ClaudeProvider(LLMProvider):
    """Anthropic Claude via official SDK."""

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self.api_key)

    def chat(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: 4 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add align/llm_fallback.py tests/test_llm_fallback.py
git commit -m "feat(align): add LLMProvider abstraction with DeepSeek and Claude impls"
```

---

## Task 8: LLM Fallback Refine 核心逻辑

**Files:**
- Modify: `align/llm_fallback.py`
- Modify: `tests/test_llm_fallback.py`

- [ ] **Step 1: 追加失败测试 —— refine 正常路径**

Append to `tests/test_llm_fallback.py`:

```python
from align.sequence_aligner import Segment, LineAlignment
from align.llm_fallback import refine, _find_low_confidence_windows


def test_find_low_confidence_windows_single():
    confidences = [0.9, 0.9, 0.4, 0.3, 0.9, 0.9]
    windows = _find_low_confidence_windows(confidences, threshold=0.55)
    assert windows == [(2, 3)]


def test_find_low_confidence_windows_multiple():
    confidences = [0.9, 0.3, 0.9, 0.4, 0.3, 0.9]
    windows = _find_low_confidence_windows(confidences, threshold=0.55)
    assert windows == [(1, 1), (3, 4)]


def test_find_low_confidence_windows_none():
    confidences = [0.9, 0.8, 0.7, 0.6]
    windows = _find_low_confidence_windows(confidences, threshold=0.55)
    assert windows == []


def test_refine_applies_llm_mapping():
    lyrics = ["行一", "行二", "行三"]
    segments = [
        Segment("行一", 0.0, 1.0),
        Segment("嗯", 1.0, 1.2),
        Segment("行二错的", 1.2, 2.5),  # 经典算法会对错这行
        Segment("行三", 2.5, 4.0),
    ]
    # 经典结果：第二行匹配到错的 segment，置信度低
    alignments = [
        LineAlignment(0, [0]),
        LineAlignment(1, [2]),  # 错误
        LineAlignment(2, [3]),
    ]
    confidences = [1.0, 0.3, 1.0]  # 第二行低置信度

    fake_provider = MagicMock(spec=LLMProvider)
    fake_provider.chat.return_value = (
        '{"alignment": ['
        '{"user_idx": 1, "segment_idxs": [2]}'
        ']}'
    )

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    assert fake_provider.chat.called
    # LLM 返回与原始一致，结果应保持
    assert refined[1].segment_idxs == [2]


def test_refine_skips_when_all_high_confidence():
    lyrics = ["行一"]
    segments = [Segment("行一", 0.0, 1.0)]
    alignments = [LineAlignment(0, [0])]
    confidences = [1.0]

    fake_provider = MagicMock(spec=LLMProvider)
    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    assert not fake_provider.chat.called
    assert refined == alignments
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: 新测试 FAIL with `ImportError: cannot import name 'refine'`

- [ ] **Step 3: 实现 refine 和 window 检测**

Append to `align/llm_fallback.py`:

```python
import json
import logging

from align.sequence_aligner import Segment, LineAlignment

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.55
CONTEXT_EXPAND = 1  # lines of anchor context on each side of a window

SYSTEM_PROMPT = """你是一个歌词对齐助手。

用户会给你两份数据：
1. user_lyrics：用户提供的标准歌词（文本正确）
2. whisper_segments：语音识别结果（带准确时间戳，但文本可能和歌词略有出入，可能含衬词"嗯啊哦"或和声）

你的任务：为每一行 user_lyrics 找到对应的 whisper_segments 索引（可以是 0 个、1 个或多个连续索引）。

严格要求：
- 只输出 JSON，格式必须为 {"alignment": [{"user_idx": N, "segment_idxs": [...]}, ...]}
- user_idx 必须按输入顺序单调递增，不得遗漏任何一行 user_lyrics
- segment_idxs 可为空数组 [] 表示这行对应间奏或 whisper 没识别到
- 不要解释，不要 markdown，只输出 JSON 对象"""

FEW_SHOT_EXAMPLE = """示例输入：
{"user_lyrics":[{"idx":0,"text":"你曾说过海枯石烂"},{"idx":1,"text":"如今却各自天涯"}],"whisper_segments":[{"idx":10,"start":5.0,"end":7.5,"text":"你曾说过海枯石烂"},{"idx":11,"start":7.5,"end":7.8,"text":"嗯"},{"idx":12,"start":7.8,"end":10.0,"text":"如今各自天涯"}]}

示例输出：
{"alignment":[{"user_idx":0,"segment_idxs":[10]},{"user_idx":1,"segment_idxs":[12]}]}"""


def _find_low_confidence_windows(
    confidences: list[float],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[tuple[int, int]]:
    """Find contiguous runs of line indices where confidence < threshold.

    Returns list of (start_idx, end_idx) inclusive ranges.
    """
    windows: list[tuple[int, int]] = []
    i = 0
    n = len(confidences)
    while i < n:
        if confidences[i] < threshold:
            start = i
            while i < n and confidences[i] < threshold:
                i += 1
            windows.append((start, i - 1))
        else:
            i += 1
    return windows


def _expand_window(
    window: tuple[int, int],
    n_lyrics: int,
    alignments: list[LineAlignment],
    n_segments: int,
    expand: int = CONTEXT_EXPAND,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Expand a window with context anchor lines on both sides, return
    ((lyric_start, lyric_end), (segment_start, segment_end)) inclusive.
    """
    l_start = max(0, window[0] - expand)
    l_end = min(n_lyrics - 1, window[1] + expand)

    # Segment range: smallest/largest segment idx referenced by lines in [l_start..l_end]
    seg_idxs: list[int] = []
    for line_idx in range(l_start, l_end + 1):
        seg_idxs.extend(alignments[line_idx].segment_idxs)
    if seg_idxs:
        s_start = max(0, min(seg_idxs) - expand)
        s_end = min(n_segments - 1, max(seg_idxs) + expand)
    else:
        # Fall back to rough proportional guess
        s_start = 0
        s_end = n_segments - 1

    return (l_start, l_end), (s_start, s_end)


def refine(
    lyric_lines: list[str],
    segments: list[Segment],
    alignments: list[LineAlignment],
    confidences: list[float],
    provider: LLMProvider,
) -> list[LineAlignment]:
    """Refine low-confidence alignment windows using an LLM.

    Returns a new alignment list. Any exception falls back to the
    original `alignments` for the affected window — this function never
    makes things worse than the classical result.
    """
    windows = _find_low_confidence_windows(confidences)
    if not windows:
        return alignments

    result = [LineAlignment(a.line_idx, list(a.segment_idxs)) for a in alignments]

    for window in windows:
        try:
            (l_start, l_end), (s_start, s_end) = _expand_window(
                window, len(lyric_lines), alignments, len(segments)
            )
            user_payload = _build_payload(lyric_lines, segments, l_start, l_end, s_start, s_end)
            mapping = _chat_and_parse_with_retry(
                provider, user_payload, l_start, l_end, s_start, s_end
            )
            for entry in mapping:
                result[entry['user_idx']].segment_idxs = entry['segment_idxs']
            logger.info(f"LLM refined window lines {l_start}-{l_end}")
        except Exception as e:
            logger.warning(f"LLM refine failed for window {window}: {e}, keeping classical result")

    return result


def _build_payload(
    lyric_lines: list[str],
    segments: list[Segment],
    l_start: int,
    l_end: int,
    s_start: int,
    s_end: int,
) -> str:
    payload = {
        "user_lyrics": [
            {"idx": i, "text": lyric_lines[i]}
            for i in range(l_start, l_end + 1)
        ],
        "whisper_segments": [
            {"idx": j, "start": round(segments[j].start, 2),
             "end": round(segments[j].end, 2), "text": segments[j].text}
            for j in range(s_start, s_end + 1)
        ],
    }
    return FEW_SHOT_EXAMPLE + "\n\n实际输入：\n" + json.dumps(payload, ensure_ascii=False)


def _chat_and_parse_with_retry(
    provider: LLMProvider,
    user_payload: str,
    l_start: int,
    l_end: int,
    s_start: int,
    s_end: int,
    retries: int = 1,
) -> list[dict]:
    """Call provider and parse response, retrying on any failure (network or parse)."""
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = provider.chat(system=SYSTEM_PROMPT, user=user_payload)
            return _parse_response(response, l_start, l_end, s_start, s_end)
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore


def _parse_response(
    response: str,
    l_start: int,
    l_end: int,
    s_start: int,
    s_end: int,
) -> list[dict]:
    data = json.loads(response)
    alignment = data.get("alignment", [])
    if not isinstance(alignment, list):
        raise ValueError("alignment field is not a list")

    last_user_idx = -1
    for entry in alignment:
        user_idx = entry["user_idx"]
        segment_idxs = entry["segment_idxs"]
        if not (l_start <= user_idx <= l_end):
            raise ValueError(f"user_idx {user_idx} out of window [{l_start},{l_end}]")
        if user_idx <= last_user_idx:
            raise ValueError("user_idx is not monotonically increasing")
        last_user_idx = user_idx
        if not isinstance(segment_idxs, list):
            raise ValueError("segment_idxs is not a list")
        for s in segment_idxs:
            if not (s_start <= s <= s_end):
                raise ValueError(f"segment_idx {s} out of window [{s_start},{s_end}]")
    return alignment
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: 全部 9 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add align/llm_fallback.py tests/test_llm_fallback.py
git commit -m "feat(align): implement LLM refine() for low-confidence windows"
```

---

## Task 9: LLM Fallback 错误降级路径

**Files:**
- Modify: `tests/test_llm_fallback.py`

- [ ] **Step 1: 追加降级路径测试**

Append to `tests/test_llm_fallback.py`:

```python
def test_refine_falls_back_on_invalid_json():
    lyrics = ["行一", "行二"]
    segments = [Segment("行一", 0.0, 1.0), Segment("对不上", 1.0, 2.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1])]
    confidences = [1.0, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    fake_provider.chat.return_value = "this is not json"

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    # 应该保留原始结果
    assert refined[1].segment_idxs == [1]


def test_refine_falls_back_on_provider_exception():
    lyrics = ["行一", "行二"]
    segments = [Segment("行一", 0.0, 1.0), Segment("对不上", 1.0, 2.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1])]
    confidences = [1.0, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    fake_provider.chat.side_effect = ConnectionError("network down")

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    # 重试 1 次后仍失败，应保留原始结果
    assert refined[1].segment_idxs == [1]
    assert fake_provider.chat.call_count == 2  # 1 次 + 1 次重试


def test_refine_rejects_non_monotonic_user_idx():
    lyrics = ["行一", "行二", "行三"]
    segments = [Segment("行一", 0.0, 1.0), Segment("X", 1.0, 2.0), Segment("Y", 2.0, 3.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1]), LineAlignment(2, [2])]
    confidences = [1.0, 0.3, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    # 返回乱序的 user_idx
    fake_provider.chat.return_value = (
        '{"alignment":['
        '{"user_idx":2,"segment_idxs":[2]},'
        '{"user_idx":1,"segment_idxs":[1]}'
        ']}'
    )

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    # 被拒绝，保留原始结果
    assert refined[1].segment_idxs == [1]
    assert refined[2].segment_idxs == [2]


def test_refine_rejects_out_of_range_segment_idx():
    lyrics = ["行一", "行二"]
    segments = [Segment("行一", 0.0, 1.0), Segment("行二", 1.0, 2.0)]
    alignments = [LineAlignment(0, [0]), LineAlignment(1, [1])]
    confidences = [1.0, 0.3]

    fake_provider = MagicMock(spec=LLMProvider)
    # segment_idx 越界
    fake_provider.chat.return_value = '{"alignment":[{"user_idx":1,"segment_idxs":[99]}]}'

    refined = refine(lyrics, segments, alignments, confidences, provider=fake_provider)

    assert refined[1].segment_idxs == [1]
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_llm_fallback.py -v`
Expected: 全部 PASS（降级逻辑已经在 Task 8 里通过 try/except 实现了）。

- [ ] **Step 3: Commit**

```bash
git add tests/test_llm_fallback.py
git commit -m "test(align): cover LLM fallback degradation paths"
```

---

## Task 10: 编排层 —— 重写 `align_lyrics`

**Files:**
- Modify: `align/lyrics_aligner.py`

- [ ] **Step 1: 读取当前 `align/lyrics_aligner.py` 确认保留的部分**

Run: `cat align/lyrics_aligner.py`
Note: 保留 `DEFAULT_MODEL`、`_detect_language`、`_even_distribute` 作为后备路径。

- [ ] **Step 2: 完全重写 `align/lyrics_aligner.py`**

Replace the entire contents of `align/lyrics_aligner.py` with:

```python
"""Orchestrate lyric alignment: Whisper transcribe → DP align → LLM refine."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mlx_whisper

from config import TimedLine
from align import sequence_aligner, llm_fallback
from align.sequence_aligner import Segment, LineAlignment
from align.llm_fallback import LLMProvider, DeepSeekProvider, ClaudeProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def _detect_language(lyrics_lines: list[str]) -> str:
    text = "".join(lyrics_lines)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "zh"
    if chinese_chars / total_alpha > 0.3:
        return "zh"
    return "en"


def _whisper_to_segments(result: dict) -> list[Segment]:
    segments: list[Segment] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if text and end > start:
            segments.append(Segment(text=text, start=start, end=end))
    return segments


def _llm_enabled() -> bool:
    return os.environ.get("LYRICS_ALIGNER_LLM_ENABLED", "true").lower() != "false"


def _get_provider() -> LLMProvider | None:
    try:
        name = os.environ.get("LYRICS_ALIGNER_LLM_PROVIDER", "deepseek").lower()
        if name == "claude":
            return ClaudeProvider()
        return DeepSeekProvider()
    except Exception as e:
        logger.warning(f"LLM provider unavailable ({e}), skipping fallback")
        return None


def _emit_timed_lines(
    lyric_lines: list[str],
    alignments: list[LineAlignment],
    segments: list[Segment],
    total_duration: float,
) -> list[TimedLine]:
    """Convert alignments to TimedLine list, filling gaps for unmatched lines."""
    timed: list[TimedLine] = []
    for i, line in enumerate(lyric_lines):
        a = alignments[i]
        if a.segment_idxs:
            start = min(segments[k].start for k in a.segment_idxs)
            end = max(segments[k].end for k in a.segment_idxs)
        else:
            # Unmatched line: place it in the gap before the next matched line,
            # or at the end if nothing follows.
            next_start = _next_matched_start(i, alignments, segments, total_duration)
            prev_end = timed[-1].end if timed else 0.0
            start = prev_end
            end = min(next_start, prev_end + 0.1)
        timed.append(TimedLine(text=line, start=start, end=end))
    return timed


def _next_matched_start(
    after_idx: int,
    alignments: list[LineAlignment],
    segments: list[Segment],
    total_duration: float,
) -> float:
    for j in range(after_idx + 1, len(alignments)):
        if alignments[j].segment_idxs:
            return min(segments[k].start for k in alignments[j].segment_idxs)
    return total_duration


def align_lyrics(
    audio_path: Path,
    lyrics_lines: list[str],
    language: str | None = None,
    model_repo: str = DEFAULT_MODEL,
) -> list[TimedLine]:
    """Align user-provided lyric lines to audio timeline.

    Flow:
    1. Whisper transcribes audio → Segment list (text + timing).
    2. Classical DP aligns lyric_lines to segments, produces per-line confidences.
    3. Low-confidence windows (< 0.55) are sent to LLM for semantic refinement.
    4. Time ranges are read from matched segments (LLM never invents timestamps).

    Caller must pre-clean lyrics (strip section markers / blank lines) via
    `align/lyrics_preprocessor.py` — this function assumes `lyrics_lines`
    contains only real lyric lines.
    """
    if not lyrics_lines:
        return []

    if language is None:
        language = _detect_language(lyrics_lines)

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        word_timestamps=True,
        language=language,
    )

    segments = _whisper_to_segments(result)
    total_duration = result.get("duration", 0.0)
    if not total_duration and segments:
        total_duration = segments[-1].end

    if not segments:
        logger.warning("Whisper produced no segments, falling back to even distribution")
        return _even_distribute(lyrics_lines, total_duration)

    # Classical alignment
    alignments, confidences = sequence_aligner.align(lyrics_lines, segments)

    # LLM refinement for low-confidence windows
    if _llm_enabled() and any(c < 0.55 for c in confidences):
        provider = _get_provider()
        if provider is not None:
            try:
                alignments = llm_fallback.refine(
                    lyrics_lines, segments, alignments, confidences, provider
                )
            except Exception as e:
                logger.warning(f"LLM fallback crashed ({e}), using classical result")

    return _emit_timed_lines(lyrics_lines, alignments, segments, total_duration)


def _even_distribute(lyrics_lines: list[str], duration: float) -> list[TimedLine]:
    if not lyrics_lines:
        return []
    line_duration = duration / len(lyrics_lines) if duration > 0 else 1.0
    return [
        TimedLine(text=line, start=i * line_duration, end=(i + 1) * line_duration)
        for i, line in enumerate(lyrics_lines)
    ]
```

- [ ] **Step 3: 快速导入检查**

Run: `python -c "from align.lyrics_aligner import align_lyrics; print('OK')"`
Expected: `OK` printed with no errors.

- [ ] **Step 4: 运行所有现有单元测试确保没破坏**

Run: `pytest tests/ -v --ignore=tests/test_alignment_e2e.py`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add align/lyrics_aligner.py
git commit -m "refactor(align): rewrite align_lyrics using sequence alignment + LLM fallback"
```

---

## Task 11: 端到端 smoke test（真实音频）

**Files:**
- Create: `tests/test_alignment_e2e.py`

- [ ] **Step 1: 写 smoke test**

Create `tests/test_alignment_e2e.py`:

```python
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
```

- [ ] **Step 2: 运行 smoke test**

Run: `pytest tests/test_alignment_e2e.py -v -s`
Expected: test PASSES and prints the full alignment. First run will be slow (model load + Whisper transcribe + LLM refine).

- [ ] **Step 3: 人工 review 打印出来的对齐结果**

打开 `sample/这温暖的家.mp3` 播放，对照终端输出的 `[start -> end] text`，记录：
- 有几行对得不错
- 有几行还有可见偏差（> 500ms）
- 有没有严重错位的行（> 3s 或张冠李戴）

如果对齐质量不达标，在下一轮迭代调整阈值或 prompt。本 task 的目标是**跑通端到端、产出可评估的结果**，不一定要第一次就完美。

- [ ] **Step 4: Commit**

```bash
git add tests/test_alignment_e2e.py
git commit -m "test(align): add e2e smoke test on sample audio"
```

---

## Task 12: 验证与调优

**Files:** 取决于发现的问题

- [ ] **Step 1: 对照真机结果判断是否达标**

成功标准（来自 spec）：
- 行级平均偏差 < 500 ms
- 行级最大偏差 < 1.5 s
- 严重偏差（> 3 s）行数 = 0

如果达标 → 跳到 Step 4。

- [ ] **Step 2: 如果达标未达标：诊断**

常见问题与调整方向：

| 观察到的症状 | 可能原因 | 调整 |
|---|---|---|
| LLM 从不被触发但结果有偏差 | 阈值 0.55 太低 | 在 `llm_fallback.py` 把 `CONFIDENCE_THRESHOLD` 改高到 0.65 |
| LLM 被频繁触发但没改善 | Prompt 不够清晰或 few-shot 不贴合 | 改写 SYSTEM_PROMPT / FEW_SHOT_EXAMPLE |
| 衬词被错误匹配进歌词行 | skip_cost 太贵 | 把 `-0.10` 改成 `-0.05`（扩大 FILLERS 集合） |
| 长句被切成两段但没合并 | 1:2 合并逻辑未生效 | 加 print 看 DP 选了什么动作 |
| 整体前后偏移 | Whisper 时间戳本身偏移 | 这是 Whisper 问题，本计划范围外 |

- [ ] **Step 3: 针对问题迭代**

做出调整后，重新运行 `pytest tests/test_alignment_e2e.py -v -s` 对比。**每次迭代都提交一个 commit**，不要一次改一堆。

- [ ] **Step 4: 达标后做最终 commit**

```bash
git add -A
git commit -m "tune(align): adjust threshold/costs for sample song validation"
```

（只有真的改动了常量才需要这一步。）

- [ ] **Step 5: 在 plan 文件底部记录最终指标**

在本 plan 文档末尾追加：

```markdown
## Validation results (2026-04-XX)

- Sample: `sample/这温暖的家.mp3`
- Avg offset: XXX ms
- Max offset: XXX ms
- Lines with severe offset (>3s): X
- LLM triggered: X windows
```

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-04-11-ai-lyrics-alignment.md
git commit -m "docs: record alignment validation results"
```
