"""Centralized font loading with fallback chain for Chinese text."""
from PIL import ImageFont
from functools import lru_cache

_FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


@lru_cache(maxsize=16)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a Chinese-capable font at the given size, with fallback."""
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
