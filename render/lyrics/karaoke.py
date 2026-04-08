# render/lyrics/karaoke.py
from PIL import Image, ImageDraw
from config import TimedLine
from render.fonts import get_font


class KaraokeLyrics:
    """KTV-style lyrics: show 3-5 lines, highlight current, smooth scroll."""

    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        if not lines:
            return

        draw = ImageDraw.Draw(frame)
        font_highlight = get_font(40, "bold")
        font_normal = get_font(30)

        current_idx = 0
        for i, line in enumerate(lines):
            if line.start <= time_s:
                current_idx = i
            if line.start > time_s:
                break

        visible_range = 2
        lyrics_area_y = int(h * 0.62)
        line_spacing = 55

        for offset in range(-visible_range, visible_range + 1):
            idx = current_idx + offset
            if idx < 0 or idx >= len(lines):
                continue

            y = lyrics_area_y + offset * line_spacing

            if current_idx < len(lines):
                cur = lines[current_idx]
                if cur.end > cur.start:
                    progress = min(1.0, (time_s - cur.start) / (cur.end - cur.start))
                    y -= int(progress * line_spacing * 0.3)

            if offset == 0:
                color = theme.get_lyrics_highlight_color()
                draw.text((w // 2, y), lines[idx].text, fill=color, anchor="mt", font=font_highlight)
            else:
                dim = theme.get_lyrics_dim_color()
                dist = abs(offset)
                fade = max(0.3, 1.0 - dist * 0.25)
                color = tuple(int(c * fade) for c in dim)
                draw.text((w // 2, y), lines[idx].text, fill=color, anchor="mt", font=font_normal)
