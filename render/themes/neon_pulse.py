# render/themes/neon_pulse.py
import math
import random
from PIL import Image, ImageDraw
from config import AudioFeatures
from render.fonts import get_font


class NeonPulseTheme:
    """Cyberpunk/neon themed visuals with glowing effects."""

    def __init__(self):
        self._particles: list[dict] | None = None
        self._gradient_cache: Image.Image | None = None

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        """Deep blue-to-purple gradient with floating particles."""
        # Cache the gradient - it never changes
        if self._gradient_cache is None or self._gradient_cache.size != (w, h):
            img = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(img)
            for y in range(h):
                ratio = y / h
                r = int(8 + 25 * ratio)
                g = int(2 + 8 * (1 - ratio))
                b = int(35 + 70 * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))
            self._gradient_cache = img

        img = self._gradient_cache.copy()
        draw = ImageDraw.Draw(img)

        if self._particles is None:
            self._particles = [
                {
                    "x": random.randint(0, w),
                    "y": random.randint(0, h),
                    "speed": random.uniform(0.3, 1.5),
                    "size": random.randint(1, 3),
                    "phase": random.uniform(0, math.pi * 2),
                }
                for _ in range(40)
            ]

        for p in self._particles:
            px = int(p["x"] + math.sin(frame_idx * 0.02 + p["phase"]) * 20)
            py = int((p["y"] - frame_idx * p["speed"]) % h)
            draw.ellipse(
                (px - p["size"], py - p["size"], px + p["size"], py + p["size"]),
                fill=(100, 200, 255),
            )

        return img

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        """Draw neon glow ring around the disc that pulses with beat."""
        draw = ImageDraw.Draw(frame)
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 2

        rms = features.rms[audio_frame] if audio_frame < len(features.rms) else 0.5
        is_beat = audio_frame in features.beat_frames

        glow_width = int(4 + 8 * rms)
        if is_beat:
            glow_width += 6

        for i in range(glow_width, 0, -1):
            phase = (audio_frame * 0.03) % (math.pi * 2)
            r = int(80 + 175 * max(0, math.sin(phase)))
            g = int(200 * max(0, math.sin(phase + math.pi * 2 / 3)))
            b = int(150 + 105 * max(0, math.sin(phase + math.pi * 4 / 3)))
            color = (r, g, b)

            ring_r = radius + 5 + i
            draw.ellipse(
                (center_x - ring_r, center_y - ring_r, center_x + ring_r, center_y + ring_r),
                outline=color, width=1,
            )

    def draw_visualizer(
        self, frame: Image.Image, audio_frame: int,
        features: AudioFeatures, w: int, h: int,
    ) -> None:
        """Draw 300 circular spectrum bars around the disc area."""
        draw = ImageDraw.Draw(frame)
        center_x = w // 2
        center_y = h // 4 + (min(w, h) // 3) // 2

        if audio_frame >= len(features.spectrum):
            return

        spectrum = features.spectrum[audio_frame]
        num_bars = 300
        base_radius = min(w, h) // 3 // 2 + 20

        for i in range(num_bars):
            # Interpolate spectrum value from 16 bands to 300 bars
            band_pos = (i / num_bars) * len(spectrum)
            band_idx = int(band_pos)
            band_frac = band_pos - band_idx
            val = spectrum[band_idx % len(spectrum)]
            if band_idx + 1 < len(spectrum):
                val = val * (1 - band_frac) + spectrum[band_idx + 1] * band_frac

            angle = (i / num_bars) * math.pi * 2 - math.pi / 2
            bar_length = int(8 + 60 * val)

            x1 = center_x + int(base_radius * math.cos(angle))
            y1 = center_y + int(base_radius * math.sin(angle))
            x2 = center_x + int((base_radius + bar_length) * math.cos(angle))
            y2 = center_y + int((base_radius + bar_length) * math.sin(angle))

            phase = (audio_frame * 0.03 + i * 0.02) % (math.pi * 2)
            r = int(80 + 175 * max(0, math.sin(phase)))
            g = int(200 * max(0, math.sin(phase + math.pi * 2 / 3)))
            b = int(150 + 105 * max(0, math.sin(phase + math.pi * 4 / 3)))

            draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=1)

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt", font=get_font(48))

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        draw.text((canvas_w // 2, y), text, fill=(180, 180, 200), anchor="mt", font=get_font(32))

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (255, 255, 255)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (0, 255, 255)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (100, 100, 120)
