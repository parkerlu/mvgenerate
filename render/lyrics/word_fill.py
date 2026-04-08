# render/lyrics/word_fill.py
from PIL import Image, ImageDraw
from config import TimedLine
from render.fonts import get_font


class WordFillLyrics:
    """Apple Music style: words fill with color as they're sung."""

    def draw(self, frame: Image.Image, time_s: float, lines: list[TimedLine], theme, w: int, h: int):
        if not lines:
            return

        draw = ImageDraw.Draw(frame)
        font = get_font(38, "bold")

        lyrics_y = int(h * 0.68)

        current = None
        for line in lines:
            if line.start <= time_s <= line.end:
                current = line
                break

        if current is None:
            return

        highlight_color = theme.get_lyrics_highlight_color()
        dim_color = theme.get_lyrics_dim_color()

        if not current.words:
            # No word-level timestamps — fallback: proportional fill
            progress = (time_s - current.start) / max(0.01, current.end - current.start)
            text = current.text
            fill_chars = int(len(text) * progress)

            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            start_x = (w - text_w) // 2

            filled = text[:fill_chars]
            remaining = text[fill_chars:]

            if filled:
                draw.text((start_x, lyrics_y), filled, fill=highlight_color, anchor="lt", font=font)
                filled_w = font.getbbox(filled)[2] - font.getbbox(filled)[0]
            else:
                filled_w = 0

            if remaining:
                draw.text((start_x + filled_w, lyrics_y), remaining, fill=dim_color, anchor="lt", font=font)
        else:
            # Word-level fill
            text = current.text
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            start_x = (w - text_w) // 2

            x_cursor = start_x
            for word in current.words:
                word_text = word.text
                if time_s >= word.end:
                    color = highlight_color
                elif time_s >= word.start:
                    progress = (time_s - word.start) / max(0.01, word.end - word.start)
                    color = tuple(
                        int(d + (h_ - d) * progress)
                        for d, h_ in zip(dim_color, highlight_color)
                    )
                else:
                    color = dim_color

                draw.text((x_cursor, lyrics_y), word_text, fill=color, anchor="lt", font=font)
                word_bbox = font.getbbox(word_text)
                x_cursor += word_bbox[2] - word_bbox[0]
