# render/themes/wave_groove.py
import math
from PIL import Image, ImageDraw, ImageFont
from config import AudioFeatures


class WaveGrooveTheme:
    """Dynamic theme with flowing wave textures and breathing effects."""

    def __init__(self):
        self._gradient_cache: Image.Image | None = None

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        # Cache the static gradient portion
        if self._gradient_cache is None or self._gradient_cache.size != (w, h):
            img = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(img)
            for y in range(h):
                ratio = y / h
                r = int(15 + 10 * ratio)
                g = int(12 + 15 * ratio)
                b = int(25 + 20 * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))
            self._gradient_cache = img

        img = self._gradient_cache.copy()
        draw = ImageDraw.Draw(img)

        rms = 0.5
        if frame_idx < len(features.rms):
            rms = features.rms[frame_idx]

        # Only draw animated wave lines (much cheaper than per-pixel gradient)
        for wave_y_base in range(0, h, 120):
            points = []
            for x in range(0, w, 4):
                y_offset = math.sin(x * 0.01 + frame_idx * 0.03 + wave_y_base * 0.1) * (10 + 20 * rms)
                points.append((x, wave_y_base + y_offset))
            if len(points) > 1:
                draw.line(points, fill=(40, 50, 80), width=1)

        return img

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        draw = ImageDraw.Draw(frame)
        cx = x + size // 2
        cy = y + size // 2
        rms = features.rms[audio_frame] if audio_frame < len(features.rms) else 0.5
        ring_r = size // 2 + 5 + int(8 * rms)
        draw.ellipse(
            (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
            outline=(60, 80, 140), width=2,
        )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        draw = ImageDraw.Draw(frame)
        cx = w // 2
        cy = h // 4 + (min(w, h) // 3) // 2

        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        base_r = min(w, h) // 3 // 2 + 20

        points = []
        num_points = 64
        for i in range(num_points + 1):
            angle = (i / num_points) * math.pi * 2
            band_idx = int((i / num_points) * len(spectrum)) % len(spectrum)
            wave = spectrum[band_idx] * 40
            r = base_r + wave + math.sin(angle * 3 + audio_frame * 0.05) * 10
            px = cx + int(r * math.cos(angle))
            py = cy + int(r * math.sin(angle))
            points.append((px, py))

        if len(points) > 1:
            draw.line(points, fill=(80, 120, 200), width=2)

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt", font=font)

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, y), text, fill=(160, 180, 220), anchor="mt", font=font)

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (255, 255, 255)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (120, 180, 255)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (80, 90, 110)
