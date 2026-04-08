# render/base.py
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import math

from config import (
    AspectRatio, Theme, LyricsStyle,
    AudioFeatures, TimedLine,
    RESOLUTIONS, FPS,
    COVER_DURATION, TRANSITION_DURATION, OUTRO_DURATION,
)


class Renderer:
    """Orchestrates per-frame rendering by compositing layers."""

    def __init__(
        self,
        cover_path: Path,
        aspect: AspectRatio,
        theme: Theme,
        lyrics_style: LyricsStyle,
        title: str = "",
        artist: str = "",
    ):
        self.width, self.height = RESOLUTIONS[aspect]
        self.theme_name = theme
        self.lyrics_style = lyrics_style
        self.title = title
        self.artist = artist

        # Load and prepare cover image
        self.cover_original = Image.open(cover_path).convert("RGB")

        # Prepare circular disc version of cover
        disc_size = min(self.width, self.height) // 3
        self.disc_size = disc_size
        cover_resized = self.cover_original.resize((disc_size, disc_size), Image.LANCZOS)
        # Create circular mask
        mask = Image.new("L", (disc_size, disc_size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, disc_size - 1, disc_size - 1), fill=255)
        self.cover_disc = cover_resized
        self.disc_mask = mask

        # Import theme and lyrics renderers
        self._theme = _get_theme(theme)
        self._lyrics_renderer = _get_lyrics_renderer(lyrics_style)

    def render_frame(
        self,
        frame_idx: int,
        time_s: float,
        features: AudioFeatures,
        lines: list[TimedLine],
    ) -> Image.Image:
        """Render a single frame at the given time."""
        total_duration = features.duration + COVER_DURATION + TRANSITION_DURATION + OUTRO_DURATION
        playback_start = COVER_DURATION + TRANSITION_DURATION

        # Create background
        frame = self._theme.draw_background(self.width, self.height, frame_idx, features)

        if time_s < COVER_DURATION:
            # Phase 1: Cover display
            progress = time_s / COVER_DURATION
            self._draw_cover_phase(frame, progress)

        elif time_s < playback_start:
            # Phase 2: Transition (cover shrinks to disc)
            progress = (time_s - COVER_DURATION) / TRANSITION_DURATION
            self._draw_transition_phase(frame, progress, frame_idx, features)

        elif time_s < total_duration - OUTRO_DURATION:
            # Phase 3: Main playback
            audio_time = time_s - playback_start
            audio_frame = min(frame_idx, len(features.rms) - 1)
            self._draw_playback_phase(frame, audio_frame, audio_time, features, lines)

        else:
            # Phase 4: Outro fade out
            remaining = total_duration - time_s
            alpha = max(0.0, remaining / OUTRO_DURATION)
            audio_time = time_s - playback_start
            audio_frame = min(frame_idx, len(features.rms) - 1)
            self._draw_playback_phase(frame, audio_frame, audio_time, features, lines)
            # Apply fade out overlay
            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, int(255 * (1 - alpha))))
            frame.paste(Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB"))

        return frame

    def _draw_cover_phase(self, frame: Image.Image, progress: float) -> None:
        """Draw the cover display phase with fade-in."""
        cover_display_size = min(self.width, self.height) // 2
        cover_resized = self.cover_original.resize(
            (cover_display_size, cover_display_size), Image.LANCZOS
        )
        x = (self.width - cover_display_size) // 2
        y = (self.height - cover_display_size) // 3

        if progress < 1.0:
            bg_crop = frame.crop((x, y, x + cover_display_size, y + cover_display_size))
            cover_resized = Image.blend(bg_crop, cover_resized, progress)

        frame.paste(cover_resized, (x, y))

        draw = ImageDraw.Draw(frame)
        if self.title:
            text_y = y + cover_display_size + 40
            self._theme.draw_title_text(draw, self.title, self.width, text_y, self.width)
        if self.artist:
            text_y = y + cover_display_size + 100
            self._theme.draw_artist_text(draw, self.artist, self.width, text_y, self.width)

    def _draw_transition_phase(
        self, frame: Image.Image, progress: float,
        frame_idx: int, features: AudioFeatures,
    ) -> None:
        """Animate cover shrinking into disc position."""
        cover_display_size = min(self.width, self.height) // 2
        current_size = int(cover_display_size - (cover_display_size - self.disc_size) * progress)

        cover_resized = self.cover_original.resize((current_size, current_size), Image.LANCZOS)
        mask = Image.new("L", (current_size, current_size), 0)
        corner_radius = int(current_size * 0.5 * progress)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, current_size - 1, current_size - 1),
            radius=corner_radius, fill=255,
        )

        disc_x = (self.width - current_size) // 2
        start_y = (self.height - cover_display_size) // 3
        end_y = self.height // 4
        disc_y = int(start_y + (end_y - start_y) * progress)

        frame.paste(cover_resized, (disc_x, disc_y), mask)

    def _draw_playback_phase(
        self, frame: Image.Image,
        audio_frame: int, audio_time: float,
        features: AudioFeatures, lines: list[TimedLine],
    ) -> None:
        """Draw the main playback: disc + visualizer + lyrics."""
        disc_x = (self.width - self.disc_size) // 2
        disc_y = self.height // 4

        angle = (audio_time * 45) % 360
        rotated = self.cover_disc.rotate(-angle, resample=Image.BICUBIC, expand=False)
        frame.paste(rotated, (disc_x, disc_y), self.disc_mask)

        self._theme.draw_disc_effects(frame, disc_x, disc_y, self.disc_size, audio_frame, features)
        self._theme.draw_visualizer(frame, audio_frame, features, self.width, self.height)
        self._lyrics_renderer.draw(frame, audio_time, lines, self._theme, self.width, self.height)

    def total_frames(self, audio_duration: float) -> int:
        """Total number of frames for the full video."""
        total = audio_duration + COVER_DURATION + TRANSITION_DURATION + OUTRO_DURATION
        return int(total * FPS)


def _get_theme(theme: Theme):
    if theme == Theme.NEON:
        from render.themes.neon_pulse import NeonPulseTheme
        return NeonPulseTheme()
    elif theme == Theme.VINYL:
        from render.themes.vinyl_minimal import VinylMinimalTheme
        return VinylMinimalTheme()
    elif theme == Theme.WAVE:
        from render.themes.wave_groove import WaveGrooveTheme
        return WaveGrooveTheme()


def _get_lyrics_renderer(style: LyricsStyle):
    if style == LyricsStyle.KARAOKE:
        from render.lyrics.karaoke import KaraokeLyrics
        return KaraokeLyrics()
    elif style == LyricsStyle.FADE:
        from render.lyrics.fade import FadeLyrics
        return FadeLyrics()
    elif style == LyricsStyle.WORD_FILL:
        from render.lyrics.word_fill import WordFillLyrics
        return WordFillLyrics()
