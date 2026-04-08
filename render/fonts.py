"""Centralized font loading with fallback chain for Chinese text."""
from PIL import ImageFont
from functools import lru_cache

# Font paths in priority order — Source Han Sans (思源黑体) preferred
_FONT_REGULAR = [
    "/Users/chunyuanlu/Library/Fonts/SourceHanSansCN-Regular.otf",
    "/Library/Fonts/Microsoft/Microsoft Yahei.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]

_FONT_BOLD = [
    "/Users/chunyuanlu/Library/Fonts/SourceHanSansCN-Bold.otf",
    "/Library/Fonts/Microsoft/SimHei.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]

_FONT_HEAVY = [
    "/Users/chunyuanlu/Library/Fonts/SourceHanSansCN-Heavy.otf",
    "/Users/chunyuanlu/Library/Fonts/SourceHanSansCN-Bold.otf",
    "/Library/Fonts/Microsoft/SimHei.ttf",
]

_FONT_TITLE = [
    "/Users/chunyuanlu/Library/Fonts/YouSheBiaoTiHei-2.ttf",
    "/Users/chunyuanlu/Library/Fonts/SourceHanSansCN-Heavy.otf",
    "/Users/chunyuanlu/Library/Fonts/SourceHanSansCN-Bold.otf",
]


def _load_first(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=32)
def get_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    """Load a Chinese font. weight: 'regular', 'bold', 'heavy', 'title'."""
    if weight == "bold":
        return _load_first(_FONT_BOLD, size)
    elif weight == "heavy":
        return _load_first(_FONT_HEAVY, size)
    elif weight == "title":
        return _load_first(_FONT_TITLE, size)
    else:
        return _load_first(_FONT_REGULAR, size)
