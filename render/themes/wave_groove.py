# render/themes/wave_groove.py
import math
from PIL import Image, ImageDraw
from config import AudioFeatures
from render.fonts import get_font


class WaveGrooveTheme:
    """Dynamic theme with flowing waves, multi-layer rings, and particle bursts."""

    def __init__(self):
        self._gradient_cache: Image.Image | None = None

    def draw_background(self, w: int, h: int, frame_idx: int, features: AudioFeatures) -> Image.Image:
        if self._gradient_cache is None or self._gradient_cache.size != (w, h):
            img = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(img)
            for y in range(h):
                ratio = y / h
                r = int(10 + 15 * ratio)
                g = int(8 + 12 * ratio)
                b = int(25 + 30 * ratio)
                draw.line([(0, y), (w, y)], fill=(r, g, b))
            self._gradient_cache = img

        img = self._gradient_cache.copy()
        draw = ImageDraw.Draw(img)

        rms = 0.5
        if frame_idx < len(features.rms):
            rms = features.rms[frame_idx]

        # Multiple wave layers with different speeds and colors
        wave_configs = [
            {"spacing": 80, "speed": 0.02, "freq": 0.008, "color": (30, 45, 90), "amp": 15},
            {"spacing": 100, "speed": 0.035, "freq": 0.012, "color": (50, 70, 130), "amp": 20},
            {"spacing": 140, "speed": 0.015, "freq": 0.006, "color": (25, 40, 70), "amp": 12},
        ]

        for wc in wave_configs:
            for wave_y_base in range(0, h, wc["spacing"]):
                points = []
                for x in range(0, w, 3):
                    y_offset = math.sin(x * wc["freq"] + frame_idx * wc["speed"] + wave_y_base * 0.1) * (wc["amp"] + wc["amp"] * rms)
                    points.append((x, wave_y_base + y_offset))
                if len(points) > 1:
                    draw.line(points, fill=wc["color"], width=1)

        return img

    def draw_disc_effects(
        self, frame: Image.Image, x: int, y: int, size: int,
        audio_frame: int, features: AudioFeatures,
    ) -> None:
        draw = ImageDraw.Draw(frame)
        cx = x + size // 2
        cy = y + size // 2
        rms = features.rms[audio_frame] if audio_frame < len(features.rms) else 0.5
        is_beat = audio_frame in features.beat_frames

        # Multiple concentric pulsing rings
        for ring_i in range(3):
            ring_r = size // 2 + 8 + ring_i * 12 + int(10 * rms)
            if is_beat:
                ring_r += 5
            alpha_ratio = 1.0 - ring_i * 0.3
            r = int(60 * alpha_ratio)
            g = int(100 * alpha_ratio)
            b = int(200 * alpha_ratio)
            draw.ellipse(
                (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
                outline=(r, g, b), width=2,
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
        rms = features.rms[audio_frame] if audio_frame < len(features.rms) else 0.5
        base_r = min(w, h) // 3 // 2 + 30

        # Layer 1: Outer wavy ring (128 points)
        num_points = 128
        points_outer = []
        for i in range(num_points + 1):
            angle = (i / num_points) * math.pi * 2
            band_idx = int((i / num_points) * len(spectrum)) % len(spectrum)
            wave = spectrum[band_idx] * 50
            r = base_r + wave + math.sin(angle * 4 + audio_frame * 0.04) * 12
            px = cx + int(r * math.cos(angle))
            py = cy + int(r * math.sin(angle))
            points_outer.append((px, py))

        if len(points_outer) > 1:
            draw.line(points_outer, fill=(80, 130, 220), width=2)

        # Layer 2: Inner wave ring (offset phase)
        points_inner = []
        inner_r = base_r - 10
        for i in range(num_points + 1):
            angle = (i / num_points) * math.pi * 2
            band_idx = int((i / num_points) * len(spectrum)) % len(spectrum)
            wave = spectrum[band_idx] * 30
            r = inner_r + wave + math.sin(angle * 5 + audio_frame * 0.06) * 8
            px = cx + int(r * math.cos(angle))
            py = cy + int(r * math.sin(angle))
            points_inner.append((px, py))

        if len(points_inner) > 1:
            draw.line(points_inner, fill=(50, 90, 160), width=1)

        # Layer 3: Bottom horizontal waveform
        wave_y = int(h * 0.85)
        wave_points = []
        for x in range(0, w, 2):
            band_idx = int((x / w) * len(spectrum)) % len(spectrum)
            amp = spectrum[band_idx] * 30 + rms * 15
            y_val = wave_y + math.sin(x * 0.02 + audio_frame * 0.08) * amp
            wave_points.append((x, y_val))
        if len(wave_points) > 1:
            draw.line(wave_points, fill=(100, 160, 255), width=2)

        # Mirror below
        wave_points_mirror = [(x, 2 * wave_y - y) for x, y in wave_points]
        if len(wave_points_mirror) > 1:
            draw.line(wave_points_mirror, fill=(60, 100, 180), width=1)

    def draw_title_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        draw.text((canvas_w // 2, y), text, fill="white", anchor="mt", font=get_font(48))

    def draw_artist_text(self, draw: ImageDraw.Draw, text: str, canvas_w: int, y: int, max_w: int) -> None:
        draw.text((canvas_w // 2, y), text, fill=(160, 180, 220), anchor="mt", font=get_font(32))

    def get_lyrics_color(self) -> tuple[int, int, int]:
        return (255, 255, 255)

    def get_lyrics_highlight_color(self) -> tuple[int, int, int]:
        return (120, 180, 255)

    def get_lyrics_dim_color(self) -> tuple[int, int, int]:
        return (80, 90, 110)
