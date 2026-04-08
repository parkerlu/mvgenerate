#!/usr/bin/env python3
"""Generate preview thumbnails for themes and lyrics styles."""
from pathlib import Path
from PIL import Image, ImageDraw
from render.fonts import get_font

from config import AudioFeatures, TimedLine, FPS

# Simulated audio features for preview
NUM_FRAMES = 300  # 10 seconds
FEATURES = AudioFeatures(
    rms=[0.3 + 0.4 * abs(__import__('math').sin(i * 0.1)) for i in range(NUM_FRAMES)],
    spectrum=[[0.2 + 0.5 * abs(__import__('math').sin(i * 0.05 + b * 0.3)) for b in range(16)] for i in range(NUM_FRAMES)],
    beat_frames=[i for i in range(0, NUM_FRAMES, 15)],
    duration=10.0,
)

SAMPLE_LINES = [
    TimedLine(text="月光洒在窗台上", start=0.0, end=2.0),
    TimedLine(text="你的影子在摇晃", start=2.5, end=4.5),
    TimedLine(text="我们一起唱吧", start=5.0, end=7.0),
    TimedLine(text="这首歌属于夜晚", start=7.5, end=9.5),
]

PREVIEW_W, PREVIEW_H = 360, 640  # Small 9:16 preview
OUTPUT_DIR = Path("web/public/previews")


def make_cover():
    """Generate a fake cover image."""
    img = Image.new("RGB", (400, 400))
    draw = ImageDraw.Draw(img)
    # Gradient
    for y in range(400):
        for x in range(400):
            r = int(80 + 120 * (x / 400))
            g = int(40 + 80 * (y / 400))
            b = int(150 - 50 * (x / 400))
            draw.point((x, y), fill=(r, g, b))
    # Circle pattern
    for radius in range(20, 180, 30):
        draw.ellipse((200 - radius, 200 - radius, 200 + radius, 200 + radius),
                      outline=(255, 255, 255, 80), width=1)
    return img


def render_theme_preview(theme_cls, name: str, cover: Image.Image):
    """Render a single frame showing the theme during playback phase."""
    theme = theme_cls()

    # Draw background
    frame = theme.draw_background(PREVIEW_W, PREVIEW_H, 120, FEATURES)

    # Draw a disc (simplified)
    disc_size = PREVIEW_W // 3
    disc = cover.resize((disc_size, disc_size), Image.LANCZOS)
    mask = Image.new("L", (disc_size, disc_size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, disc_size - 1, disc_size - 1), fill=255)
    disc_x = (PREVIEW_W - disc_size) // 2
    disc_y = PREVIEW_H // 4
    frame.paste(disc, (disc_x, disc_y), mask)

    # Draw disc effects
    theme.draw_disc_effects(frame, disc_x, disc_y, disc_size, 120, FEATURES)

    # Draw visualizer
    theme.draw_visualizer(frame, 120, FEATURES, PREVIEW_W, PREVIEW_H)

    # Draw sample lyrics (karaoke style, simplified)
    draw = ImageDraw.Draw(frame)
    font_big = get_font(22)
    font_small = get_font(16)

    lyrics_y = int(PREVIEW_H * 0.62)
    highlight = theme.get_lyrics_highlight_color()
    dim = theme.get_lyrics_dim_color()

    draw.text((PREVIEW_W // 2, lyrics_y - 35), SAMPLE_LINES[0].text, fill=dim, anchor="mt", font=font_small)
    draw.text((PREVIEW_W // 2, lyrics_y), SAMPLE_LINES[1].text, fill=highlight, anchor="mt", font=font_big)
    draw.text((PREVIEW_W // 2, lyrics_y + 35), SAMPLE_LINES[2].text, fill=dim, anchor="mt", font=font_small)

    frame.save(OUTPUT_DIR / f"theme_{name}.png")
    print(f"Saved theme_{name}.png")


def render_lyrics_preview(style_name: str):
    """Render a preview showing each lyrics display style."""
    w, h = 360, 200

    img = Image.new("RGB", (w, h), (15, 15, 30))
    draw = ImageDraw.Draw(img)

    font_big = get_font(22)
    font_med = get_font(18)
    font_small = get_font(16)

    highlight = (0, 255, 255)
    dim = (80, 80, 100)
    mid = (40, 180, 180)

    if style_name == "karaoke":
        # Show multiple lines, middle one highlighted
        lines = ["月光洒在窗台上", "你的影子在摇晃", "我们一起唱吧", "这首歌属于夜晚"]
        y_start = 30
        spacing = 40
        for i, line in enumerate(lines):
            y = y_start + i * spacing
            if i == 1:  # highlighted
                draw.text((w // 2, y), line, fill=highlight, anchor="mt", font=font_big)
                # Draw a subtle highlight bar behind
                bbox = font_big.getbbox(line)
                tw = bbox[2] - bbox[0]
                draw.rectangle(
                    (w // 2 - tw // 2 - 8, y - 14, w // 2 + tw // 2 + 8, y + 18),
                    fill=(0, 40, 40),
                )
                draw.text((w // 2, y), line, fill=highlight, anchor="mt", font=font_big)
            else:
                fade = max(0.4, 1.0 - abs(i - 1) * 0.3)
                color = tuple(int(c * fade) for c in dim)
                draw.text((w // 2, y), line, fill=color, anchor="mt", font=font_small)

        # Label
        draw.text((w // 2, h - 20), "KTV 逐句高亮", fill=(120, 120, 140), anchor="mt", font=font_small)

    elif style_name == "fade":
        # Show single line centered, with fade effect hint
        center_y = h // 2 - 15

        # Faded out previous line (very dim)
        draw.text((w // 2, center_y - 35), "月光洒在窗台上", fill=(30, 30, 40), anchor="mt", font=font_med)

        # Current line (bright)
        draw.text((w // 2, center_y), "你的影子在摇晃", fill=highlight, anchor="mt", font=font_big)

        # Draw fade arrows
        for i in range(5):
            alpha = int(255 * (1 - i / 5))
            y_pos = center_y - 50 - i * 4
            color = (0, alpha // 3, alpha // 3)
            draw.line([(w // 2 - 20, y_pos), (w // 2 + 20, y_pos)], fill=color, width=1)

        draw.text((w // 2, h - 20), "逐句淡入", fill=(120, 120, 140), anchor="mt", font=font_small)

    elif style_name == "word-fill":
        # Show a line being filled word by word
        center_y = h // 2 - 10
        text = "你的影子在摇晃"

        # Draw character by character with fill progress
        bbox = font_big.getbbox(text)
        text_w = bbox[2] - bbox[0]
        start_x = (w - text_w) // 2

        chars = list(text)
        x_cursor = start_x
        fill_count = 4  # First 4 chars filled

        for i, ch in enumerate(chars):
            if i < fill_count:
                color = highlight
            elif i == fill_count:
                # Partially filled - gradient color
                color = mid
            else:
                color = dim
            draw.text((x_cursor, center_y), ch, fill=color, anchor="lt", font=font_big)
            ch_w = font_big.getbbox(ch)[2] - font_big.getbbox(ch)[0]
            x_cursor += ch_w

        # Draw a small progress indicator
        progress = fill_count / len(chars)
        bar_w = text_w
        bar_y = center_y + 32
        draw.rectangle((start_x, bar_y, start_x + bar_w, bar_y + 3), fill=(30, 30, 40))
        draw.rectangle((start_x, bar_y, start_x + int(bar_w * progress), bar_y + 3), fill=highlight)

        draw.text((w // 2, h - 20), "逐字填充", fill=(120, 120, 140), anchor="mt", font=font_small)

    img.save(OUTPUT_DIR / f"lyrics_{style_name}.png")
    print(f"Saved lyrics_{style_name}.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cover = make_cover()

    # Generate theme previews
    from render.themes.neon_pulse import NeonPulseTheme
    from render.themes.vinyl_minimal import VinylMinimalTheme
    from render.themes.wave_groove import WaveGrooveTheme

    render_theme_preview(NeonPulseTheme, "neon", cover)
    render_theme_preview(VinylMinimalTheme, "vinyl", cover)
    render_theme_preview(WaveGrooveTheme, "wave", cover)

    # Generate lyrics style previews
    render_lyrics_preview("karaoke")
    render_lyrics_preview("fade")
    render_lyrics_preview("word-fill")

    print("All previews generated!")


if __name__ == "__main__":
    main()
