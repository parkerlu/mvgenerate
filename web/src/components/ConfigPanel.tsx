interface Config {
  aspect: string
  theme: string
  lyricsStyle: string
}

interface Props {
  config: Config
  onConfigChange: (config: Config) => void
}

const THEMES = [
  { value: 'neon', label: 'Neon Pulse' },
  { value: 'vinyl', label: 'Vinyl Minimal' },
  { value: 'wave', label: 'Wave Groove' },
]

const LYRICS_STYLES = [
  { value: 'karaoke', label: 'KTV Highlight' },
  { value: 'fade', label: 'Fade In/Out' },
  { value: 'word-fill', label: 'Word Fill' },
]

export default function ConfigPanel({ config, onConfigChange }: Props) {
  function update(key: keyof Config, value: string) {
    onConfigChange({ ...config, [key]: value })
  }

  return (
    <div className="section">
      <h2>Settings</h2>

      <div className="radio-group">
        <h3 style={{ fontSize: '0.85rem', color: '#888', marginBottom: 8 }}>Aspect Ratio</h3>
        {['9:16', '16:9'].map((v) => (
          <label key={v}>
            <input
              type="radio"
              name="aspect"
              value={v}
              checked={config.aspect === v}
              onChange={() => update('aspect', v)}
            />
            {v}
          </label>
        ))}
      </div>

      <div className="radio-group">
        <h3 style={{ fontSize: '0.85rem', color: '#888', marginBottom: 8 }}>Theme</h3>
        {THEMES.map((t) => (
          <label key={t.value}>
            <input
              type="radio"
              name="theme"
              value={t.value}
              checked={config.theme === t.value}
              onChange={() => update('theme', t.value)}
            />
            {t.label}
          </label>
        ))}
      </div>

      <div className="radio-group">
        <h3 style={{ fontSize: '0.85rem', color: '#888', marginBottom: 8 }}>Lyrics Style</h3>
        {LYRICS_STYLES.map((s) => (
          <label key={s.value}>
            <input
              type="radio"
              name="lyricsStyle"
              value={s.value}
              checked={config.lyricsStyle === s.value}
              onChange={() => update('lyricsStyle', s.value)}
            />
            {s.label}
          </label>
        ))}
      </div>
    </div>
  )
}
