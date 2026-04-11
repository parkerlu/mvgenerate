# AI 歌词对齐方案设计

**日期**：2026-04-11
**状态**：Draft — pending user review
**相关代码**：`align/lyrics_aligner.py`, `align/transcriber.py`, `mvgenerate.py`

## 背景与问题

当前 `align/lyrics_aligner.py::align_lyrics()` 的做法是：跑 Whisper 拿到词级时间戳 → **按字符数比例**把用户歌词行切分到这些时间戳上。

这个做法的根本缺陷是**它从不匹配文本内容**，只按比例分配。后果：

- Whisper 漏听一个字 → 后续所有行累积偏移
- 遇到间奏/前奏 → 比例全错
- 歌词里的衬词（嗯、啊）、和声、结构标记（`[Chorus]`、`[Verse1]`）都被当成普通字符计入比例，污染对齐

用户观测到的症状：歌词"有时晚几秒，有时早几秒，有时完全离谱"，约 70% 未能正确对齐。

**关键洞察**：用户已经确认 Whisper 自己的转写质量没问题（文本大致正确、时间戳精确）。问题**纯粹**出在"把用户的标准歌词映射到 Whisper 时间轴"这一步。所以这是一个**文本对齐问题**，不是 ASR 问题。

## 目标

重写 `align_lyrics()`，把问题从"按比例切"改为"**两段文本的最优序列对齐**"：

- **输入**：音频 + 用户提供的标准歌词文本（权威内容）
- **Whisper 产出**：转写文本 + 精确时间戳（权威时间）
- **输出**：每行用户歌词对应的 `TimedLine(text, start, end)`

**成功指标**（在 `sample/这温暖的家.mp3` 及后续加入的 bad case 上度量）：

| 指标 | 阈值 |
|---|---|
| 行级时间戳平均偏差 | < 500 ms |
| 行级时间戳最大偏差 | < 1.5 s |
| 严重偏差（> 3 s）行数 | 0 |

## 非目标

- 不重写 Whisper 转写逻辑 — `transcribe_lyrics()` 保持不变
- 不改变 `align_lyrics()` 的调用方签名 — `mvgenerate.py`、`server/routes.py` 零改动
- 不做无歌词的纯转写对齐（已有 `transcribe.py` 走 `transcriber.transcribe_lyrics` 路径）
- 不做多语言同时处理 — 沿用现有按歌词检测语言的逻辑

## 整体架构

```
align/
├── transcriber.py         # 已有，不动
├── lyrics_aligner.py      # 重写：入口，编排
├── sequence_aligner.py    # 新增：经典 DP 对齐（纯函数）
└── llm_fallback.py        # 新增：LLM provider 抽象 + 精修逻辑
```

**数据流**：

```
audio + user_lyrics
    ↓ transcribe_lyrics()
whisper_segments [Segment(text, start, end), ...]
    ↓ preprocess_lyrics()
tagged_lines [('lyric'|'gap'|'marker', text), ...]
    ↓ sequence_aligner.align()
preliminary_alignment + per_line_confidence
    ↓ llm_fallback.refine()        ← 仅对低置信度窗口
refined_alignment
    ↓ emit_timed_lines()            ← 剔除 marker 行
list[TimedLine]
```

每个模块独立可测：`sequence_aligner` 是纯函数（文本进、索引映射出），`llm_fallback` 可 mock provider。

## 组件详细设计

### 1. 歌词预处理 (`lyrics_aligner.preprocess_lyrics`)

区分三类行，决定它们是否参与对齐：

| 类型 | 识别规则 | 是否进 DP | 输出字幕里保留？ |
|---|---|---|---|
| `marker` | 正则 `^\s*\[.*\]\s*$`（如 `[Intro]`、`[Chorus]`、`[male vocal]`） | 否 | 否（直接剔除） |
| `gap` | 空行或纯空白 | 否（作为间奏占位） | 是（零长度或前后行的空隙） |
| `lyric` | 其他 | 是 | 是 |

**为什么剔除 marker**：它们不是唱出来的内容，没有对应时间戳；保留会污染对齐。

### 2. 经典序列对齐 (`sequence_aligner.align`)

**签名**：

```python
def align(
    lyric_lines: list[str],       # 仅 'lyric' 类型的行
    segments: list[Segment],      # Whisper 片段
) -> tuple[list[LineAlignment], list[float]]:
    """返回每行的对齐结果和置信度。"""
```

其中 `LineAlignment = (line_idx, [segment_idx, ...])`，空列表代表"这行没匹配上任何片段"。

**相似度函数**：

```python
def sim(a: str, b: str) -> float:
    a_chars = ''.join(c for c in a if c.isalnum() or '\u4e00' <= c <= '\u9fff')
    b_chars = ''.join(c for c in b if c.isalnum() or '\u4e00' <= c <= '\u9fff')
    return difflib.SequenceMatcher(None, a_chars, b_chars).ratio()
```

`SequenceMatcher` 对中文字符序列天然 work，不需要分词，对标点/空格/语气助词差异不敏感。

**DP 定义**：`dp[i][j]` = 处理完前 i 行歌词和前 j 个 Whisper 片段的最大累积得分。

**转移**：

| 动作 | 转移公式 | 说明 |
|---|---|---|
| 匹配 1:1 | `dp[i-1][j-1] + sim(L[i], W[j])` | 一行 ↔ 一段 |
| 匹配 1:2 | `dp[i-1][j-2] + sim(L[i], W[j-1]+W[j])` | 一行 ↔ 两段相邻片段（Whisper 把一句切成两段） |
| 匹配 2:1 | `dp[i-2][j-1] + sim(L[i-1]+L[i], W[j])` | 两行 ↔ 一段（一段里唱了两行） |
| 跳过歌词行 | `dp[i-1][j] - 0.5` | 为 gap 保留（间奏） |
| 跳过片段 | `dp[i][j-1] + skip_cost(W[j])` | 衬词/和声 |

**skip_cost 按片段内容动态计算**：

```python
def skip_cost(segment):
    text = strip_non_alnum_cjk(segment.text)
    FILLERS = set("嗯啊哦呀呃喔唉嘿哟噢咦哈")
    if all(c in FILLERS for c in text):
        return -0.05          # 纯衬词：几乎免费跳
    if len(text) <= 2:
        return -0.10          # 极短片段：便宜
    return -min(0.5, 0.05 * len(text))  # 正常片段：代价随长度增
```

**时间戳推导**：回溯 DP 得到每行的片段索引列表后，取 `start = min(W[k].start)`、`end = max(W[k].end)`。空映射的行用邻居的时间空隙填充（零长度可接受）。

**置信度**：每行 `confidence = sim(L[i], concat(matched_segments))`。完全未匹配的行置信度记为 `0.0`。

**复杂度**：O(N × M)，N ≈ 50、M ≈ 100，毫秒级完成。

### 3. LLM Fallback (`llm_fallback.refine`)

**触发条件**：经典对齐跑完后，收集置信度 `< 0.55` 的**连续**行，成为一个"窗口"，前后各扩 1 行作为锚点上下文。

**Provider 抽象**：

```python
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system: str, user: str) -> str: ...

class DeepSeekProvider(LLMProvider):
    """使用 deepseek-chat，OpenAI 兼容 API，key 从 DEEPSEEK_API_KEY 读取。"""

class ClaudeProvider(LLMProvider):
    """使用 claude-sonnet-4-6，key 从 ANTHROPIC_API_KEY 读取。"""
```

默认 provider = DeepSeek（成本约为 Claude 的 1/10）；通过环境变量 `LYRICS_ALIGNER_LLM_PROVIDER=claude` 切换。

**请求格式**（结构化 JSON）：

```json
{
  "user_lyrics": [
    {"idx": 11, "text": "你曾说过海枯石烂"},
    {"idx": 12, "text": "如今却各自天涯"}
  ],
  "whisper_segments": [
    {"idx": 23, "start": 45.2, "end": 48.1, "text": "你曾说过海枯石烂"},
    {"idx": 24, "start": 48.1, "end": 48.4, "text": "嗯"},
    {"idx": 25, "start": 48.4, "end": 51.0, "text": "如今各自天涯"}
  ]
}
```

**响应格式**（严格 JSON）：

```json
{
  "alignment": [
    {"user_idx": 11, "segment_idxs": [23]},
    {"user_idx": 12, "segment_idxs": [25]}
  ]
}
```

**关键约束**：LLM **只返回索引映射，绝不生成时间戳**。时间戳始终从 Whisper 片段里查。这样 LLM 的错误被限制在"哪行对哪段"这个语义问题上，不会污染时间精度。

**Prompt 结构**：

1. 系统提示：角色定位（歌词对齐助手）、输入输出规范、三条硬约束：
   - 输出必须是合法 JSON
   - `user_idx` 必须单调递增
   - `segment_idxs` 可为空（代表对应行是间奏或无法匹配）
2. 一个 few-shot 示例（覆盖衬词 + 改词场景）
3. 当前窗口 JSON

**应用结果到 preliminary**：只替换 LLM 处理过的那个窗口的行；其他行保持经典算法结果。

### 4. 编排入口 (`lyrics_aligner.align_lyrics`)

签名保持不变（向后兼容）：

```python
def align_lyrics(
    audio_path: Path,
    lyrics_lines: list[str],
    language: str | None = None,
    model_repo: str = DEFAULT_MODEL,
) -> list[TimedLine]:
    language = language or _detect_language(lyrics_lines)
    whisper_segments = transcribe_lyrics(audio_path, language, model_repo)

    tagged = list(preprocess_lyrics(lyrics_lines))
    lyric_only = [line for kind, line in tagged if kind == 'lyric']

    preliminary, confidences = sequence_aligner.align(lyric_only, whisper_segments)

    # _llm_enabled() 读取 LYRICS_ALIGNER_LLM_ENABLED 环境变量（默认 true）
    if _llm_enabled() and any(c < 0.55 for c in confidences):
        try:
            preliminary = llm_fallback.refine(lyric_only, whisper_segments, preliminary, confidences)
        except Exception as e:
            logger.warning(f"LLM fallback failed, using classical result: {e}")

    return _emit_timed_lines(tagged, preliminary, whisper_segments)
```

`_emit_timed_lines()` 负责：
- 剔除 marker 行
- 把 lyric 行的对齐结果转为 `TimedLine`
- gap 行获得零长度或邻居空隙的时间戳

## 错误处理

| 错误场景 | 行为 |
|---|---|
| Whisper 转写失败 | 抛异常（上游已有处理） |
| `sequence_aligner` 异常 | 抛异常 — 这是真 bug，不该吞 |
| LLM 无 key / 超时 / 网络失败 | 降级到经典结果，`logger.warning` |
| LLM 返回非法 JSON | 重试 1 次；再失败降级 |
| LLM 返回 `user_idx` 乱序 | 视为失败，降级 |
| LLM 返回的 `segment_idx` 越界 | 降级 |

**核心原则**：LLM fallback 只能让结果**更好**，不能让结果**更差**。任何异常路径回退到经典算法的结果。

## 测试策略

### 单元测试 `tests/test_sequence_aligner.py`

纯算法，离线可跑，不需要模型或 API：

- `sim()` 对中文、标点、空格、衬词的鲁棒性
- DP 基础场景：1:1、1:2、2:1、跳过衬词片段、跳过空歌词行
- 边界：空输入、单行输入、全部低置信度
- 结构标记识别正则

### LLM Fallback 测试 `tests/test_llm_fallback.py`

使用 mock provider：

- 合法 JSON 响应 → 正确应用映射
- 非法 JSON → 重试 1 次并降级
- 网络异常 → 降级
- `user_idx` 非单调 → 降级
- `segment_idx` 越界 → 降级

### 端到端回归 `tests/test_alignment_e2e.py`

使用 `sample/这温暖的家.mp3` + `这温暖的家.txt` 作为第一个测试样本：

- 人工标注每行的真值时间戳（保存为 `sample/这温暖的家.truth.json`）
- 指标：
  - 行级平均偏差 < 500 ms
  - 行级最大偏差 < 1.5 s
  - 严重偏差（> 3 s）行数 = 0
- 随后可加入更多 bad case，持续回归

## 集成与依赖

**新增依赖**：
- `openai`（用于 DeepSeek，OpenAI 兼容客户端）
- `anthropic`（如果启用 Claude provider）
- `python-dotenv`（读取 `.env` 文件中的 API key，如尚未引入）

**环境变量**（`.env`）：
```
DEEPSEEK_API_KEY=...
ANTHROPIC_API_KEY=...
LYRICS_ALIGNER_LLM_PROVIDER=deepseek    # or "claude"
LYRICS_ALIGNER_LLM_ENABLED=true         # 设为 false 可强制关闭 LLM fallback
```

**调用方改动**：无。`mvgenerate.py:58` 和 `server/routes.py` 的调用签名不变。

## 开放问题与后续

- 置信度阈值 `0.55` 是经验值，需要在 `这温暖的家` 真机跑完后微调
- skip_cost 里的 FILLERS 字符集目前是固定列表，若后续遇到更多衬词可扩充
- 暂不实现 word-level（词级）对齐，行级已能满足 MV 字幕需求
