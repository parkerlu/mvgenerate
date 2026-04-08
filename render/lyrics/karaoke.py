# render/lyrics/karaoke.py
from PIL import Image, ImageDraw
from config import TimedLine


class KaraokeLyrics:
    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        draw = ImageDraw.Draw(frame)
        current_idx = 0
        for i, line in enumerate(lines):
            if line.start <= time_s <= line.end:
                current_idx = i
                break
            if line.start > time_s:
                current_idx = max(0, i - 1)
                break

        visible_range = 2
        lyrics_y = int(h * 0.65)
        line_height = 50

        for offset in range(-visible_range, visible_range + 1):
            idx = current_idx + offset
            if 0 <= idx < len(lines):
                y = lyrics_y + offset * line_height
                if offset == 0:
                    color = theme.get_lyrics_highlight_color()
                else:
                    color = theme.get_lyrics_dim_color()
                draw.text((w // 2, y), lines[idx].text, fill=color, anchor="mt")
