# render/themes/vinyl_minimal.py
import math
from PIL import Image, ImageDraw, ImageFont
from config import AudioFeatures


class VinylMinimalTheme:
    """Clean, minimal theme with vinyl record aesthetic."""

    def __init__(self):
        self._gradient_cache: Image.Image | None = None

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        if self._gradient_cache is None or self._gradient_cache.size != (w, h):
            img = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(img)
            for y in range(h):
                ratio = y / h
                r = int(240 - 20 * ratio)
                g = int(235 - 25 * ratio)
                b = int(225 - 20 * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))
            self._gradient_cache = img
        return self._gradient_cache.copy()

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        draw = ImageDraw.Draw(frame)
        cx = x + size // 2
        cy = y + size // 2
        radius = size // 2

        for i in range(3, radius, 8):
            draw.ellipse(
                (cx - radius - i, cy - radius - i, cx + radius + i, cy + radius + i),
                outline=(60, 55, 50), width=1,
            )

        arm_start_x = x + size + 20
        arm_start_y = y - 10
        arm_angle = 0.7 + 0.1 * math.sin(audio_frame * 0.01)
        arm_length = size // 2 + 30
        arm_end_x = int(arm_start_x - arm_length * math.sin(arm_angle))
        arm_end_y = int(arm_start_y + arm_length * math.cos(arm_angle))
        draw.line([(arm_start_x, arm_start_y), (arm_end_x, arm_end_y)], fill=(80, 75, 70), width=3)
        draw.ellipse(
            (arm_start_x - 5, arm_start_y - 5, arm_start_x + 5, arm_start_y + 5),
            fill=(80, 75, 70),
        )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        draw = ImageDraw.Draw(frame)
        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        num_bars = len(spectrum)
        bar_width = max(2, w // (num_bars * 3))
        total_w = num_bars * bar_width * 2
        start_x = (w - total_w) // 2
        base_y = int(h * 0.88)

        for i, val in enumerate(spectrum):
            bar_h = int(5 + 60 * val)
            bx = start_x + i * bar_width * 2
            draw.rectangle(
                (bx, base_y - bar_h, bx + bar_width, base_y),
                fill=(60, 55, 50),
            )

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(40, 35, 30), anchor="mt", font=font)

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(100, 95, 90), anchor="mt", font=font)

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (40, 35, 30)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (20, 15, 10)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (160, 155, 150)
