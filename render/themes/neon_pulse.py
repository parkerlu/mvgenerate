# render/themes/neon_pulse.py
from PIL import Image, ImageDraw
from config import AudioFeatures


class NeonPulseTheme:
    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            ratio = y / h
            r = int(10 + 20 * ratio)
            g = int(5 + 10 * (1 - ratio))
            b = int(40 + 60 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        return img

    def draw_disc_effects(self, frame, x, y, size, audio_frame, features):
        pass

    def draw_visualizer(self, frame, audio_frame, features, w, h):
        pass

    def draw_title_text(self, draw, text, canvas_w, y, max_w):
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt")

    def draw_artist_text(self, draw, text, canvas_w, y, max_w):
        draw.text((canvas_w // 2, y), text, fill=(200, 200, 200), anchor="mt")

    def get_lyrics_color(self):
        return (255, 255, 255)

    def get_lyrics_highlight_color(self):
        return (0, 255, 255)

    def get_lyrics_dim_color(self):
        return (100, 100, 100)
