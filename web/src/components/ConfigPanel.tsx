interface Config {
  aspect: string
  theme: string
  lyricsStyle: string
  mode: string
}

interface Props {
  mode: 'theme' | 'lyrics'
  config: Config
  onConfigChange: (config: Config) => void
}

const THEMES = [
  {
    value: 'neon',
    label: 'Neon Pulse',
    desc: 'Cyberpunk neon glow with floating particles and circular spectrum bars',
    preview: '/previews/theme_neon.png',
  },
  {
    value: 'vinyl',
    label: 'Vinyl Minimal',
    desc: 'Clean, warm tones with vinyl record grooves and tonearm',
    preview: '/previews/theme_vinyl.png',
  },
  {
    value: 'wave',
    label: 'Wave Groove',
    desc: 'Dark flowing waves with breathing disc and circular waveform',
    preview: '/previews/theme_wave.png',
  },
]

const LYRICS_STYLES = [
  {
    value: 'karaoke',
    label: 'KTV Highlight',
    desc: 'Show multiple lines with the current line highlighted, KTV style scrolling',
    preview: '/previews/lyrics_karaoke.png',
  },
  {
    value: 'fade',
    label: 'Fade In/Out',
    desc: 'Display one line at a time with smooth fade transitions',
    preview: '/previews/lyrics_fade.png',
  },
  {
    value: 'word-fill',
    label: 'Word Fill',
    desc: 'Words light up one by one as they are sung, Apple Music style',
    preview: '/previews/lyrics_word-fill.png',
  },
]

export default function ConfigPanel({ mode, config, onConfigChange }: Props) {
  if (mode === 'theme') {
    return (
      <div className="card-grid">
        {THEMES.map((t) => (
          <div
            key={t.value}
            className={`preview-card ${config.theme === t.value ? 'selected' : ''}`}
            onClick={() => onConfigChange({ ...config, theme: t.value })}
          >
            <img src={t.preview} alt={t.label} className="preview-img theme-img" />
            <div className="preview-info">
              <strong>{t.label}</strong>
              <p>{t.desc}</p>
            </div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="card-grid lyrics-grid">
      {LYRICS_STYLES.map((s) => (
        <div
          key={s.value}
          className={`preview-card ${config.lyricsStyle === s.value ? 'selected' : ''}`}
          onClick={() => onConfigChange({ ...config, lyricsStyle: s.value })}
        >
          <img src={s.preview} alt={s.label} className="preview-img lyrics-img" />
          <div className="preview-info">
            <strong>{s.label}</strong>
            <p>{s.desc}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
