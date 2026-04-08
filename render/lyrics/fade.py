# render/lyrics/fade.py
from PIL import Image, ImageDraw
from config import TimedLine
from render.fonts import get_font


class FadeLyrics:
    """Single line display with fade in/out transitions."""

    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        if not lines:
            return

        draw = ImageDraw.Draw(frame)
        font = get_font(38, "bold")

        lyrics_y = int(h * 0.68)

        current = None
        for i, line in enumerate(lines):
            if line.start <= time_s <= line.end:
                current = line
                break

        if current is None:
            return

        fade_duration = 0.3
        if time_s - current.start < fade_duration:
            alpha = (time_s - current.start) / fade_duration
        elif current.end - time_s < fade_duration:
            alpha = (current.end - time_s) / fade_duration
        else:
            alpha = 1.0

        alpha = max(0.0, min(1.0, alpha))

        highlight = theme.get_lyrics_highlight_color()
        color = tuple(int(c * alpha) for c in highlight)
        draw.text((w // 2, lyrics_y), current.text, fill=color, anchor="mt", font=font)
