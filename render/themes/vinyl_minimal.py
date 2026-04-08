# render/themes/vinyl_minimal.py
import math
from PIL import Image, ImageDraw
from config import AudioFeatures
from render.fonts import get_font


class VinylMinimalTheme:
    """Dark minimal theme with vinyl record aesthetic and peak-hold bars."""

    def __init__(self):
        self._gradient_cache: Image.Image | None = None
        self._peak_heights: list[float] = []
        self._peak_decay = 0.95  # Peak falls slowly

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        if self._gradient_cache is None or self._gradient_cache.size != (w, h):
            img = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(img)
            for y in range(h):
                ratio = y / h
                r = int(18 + 12 * ratio)
                g = int(16 + 10 * ratio)
                b = int(22 + 15 * ratio)
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

        # Vinyl grooves
        for i in range(3, radius, 8):
            draw.ellipse(
                (cx - radius - i, cy - radius - i, cx + radius + i, cy + radius + i),
                outline=(50, 48, 45), width=1,
            )

        # Tonearm
        arm_start_x = x + size + 20
        arm_start_y = y - 10
        arm_angle = 0.7 + 0.1 * math.sin(audio_frame * 0.01)
        arm_length = size // 2 + 30
        arm_end_x = int(arm_start_x - arm_length * math.sin(arm_angle))
        arm_end_y = int(arm_start_y + arm_length * math.cos(arm_angle))
        draw.line([(arm_start_x, arm_start_y), (arm_end_x, arm_end_y)], fill=(120, 115, 100), width=3)
        draw.ellipse(
            (arm_start_x - 5, arm_start_y - 5, arm_start_x + 5, arm_start_y + 5),
            fill=(120, 115, 100),
        )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        draw = ImageDraw.Draw(frame)
        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        num_bars = 32
        bar_width = max(3, w // (num_bars * 2 + num_bars))
        gap = max(2, bar_width // 2)
        total_w = num_bars * (bar_width + gap) - gap
        start_x = (w - total_w) // 2
        base_y = int(h * 0.88)
        max_bar_h = 100

        # Initialize peak heights
        if not self._peak_heights or len(self._peak_heights) != num_bars:
            self._peak_heights = [0.0] * num_bars

        for i in range(num_bars):
            # Interpolate from spectrum bands
            band_pos = (i / num_bars) * len(spectrum)
            band_idx = int(band_pos) % len(spectrum)
            val = spectrum[band_idx]

            bar_h = int(5 + max_bar_h * val)
            bx = start_x + i * (bar_width + gap)

            # Main bar - warm amber color
            bar_color = (180, 140, 60)
            draw.rectangle(
                (bx, base_y - bar_h, bx + bar_width, base_y),
                fill=bar_color,
            )

            # Peak hold line
            if bar_h > self._peak_heights[i]:
                self._peak_heights[i] = bar_h
            else:
                self._peak_heights[i] *= self._peak_decay

            peak_y = int(self._peak_heights[i])
            if peak_y > 3:
                peak_color = (255, 220, 100)  # Bright highlight
                draw.rectangle(
                    (bx, base_y - peak_y - 3, bx + bar_width, base_y - peak_y),
                    fill=peak_color,
                )

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        draw.text((canvas_w // 2, y), text, fill=(220, 210, 190), anchor="mt", font=get_font(48))

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        draw.text((canvas_w // 2, y), text, fill=(150, 140, 120), anchor="mt", font=get_font(32))

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (200, 190, 170)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (255, 220, 100)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (80, 75, 65)
