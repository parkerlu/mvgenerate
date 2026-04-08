import { useState } from 'react'
import UploadArea from './components/UploadArea'
import ConfigPanel from './components/ConfigPanel'
import ProgressBar from './components/ProgressBar'
import ResultView from './components/ResultView'

interface UploadedFiles {
  audio?: { path: string; name: string }
  lyrics?: { path: string; name: string }
  cover?: { path: string; name: string; previewUrl?: string }
}

interface Config {
  aspect: string
  theme: string
  lyricsStyle: string
  mode: string
}

const TOTAL_STEPS = 5

export default function App() {
  const [step, setStep] = useState(1)
  const [files, setFiles] = useState<UploadedFiles>({})
  const [config, setConfig] = useState<Config>({
    aspect: '9:16',
    theme: 'neon',
    lyricsStyle: 'karaoke',
    mode: 'full',
  })
  const [title, setTitle] = useState(() => localStorage.getItem('mv_title') || '')
  const [artist, setArtist] = useState(() => localStorage.getItem('mv_artist') || '')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)

  const canGoNext = () => {
    if (step === 1) return !!(files.audio && files.lyrics && files.cover)
    return true
  }

  async function handleGenerate() {
    if (!files.audio || !files.lyrics || !files.cover) return

    setIsGenerating(true)
    setStep(TOTAL_STEPS)
    setProgress(0)
    setStatusMsg('Starting...')
    setResultUrl(null)

    const formData = new FormData()
    formData.append('audio_path', files.audio.path)
    formData.append('lyrics_path', files.lyrics.path)
    formData.append('cover_path', files.cover.path)
    formData.append('aspect', config.aspect)
    formData.append('theme', config.theme)
    formData.append('lyrics_style', config.lyricsStyle)
    formData.append('mode', config.mode)
    formData.append('title', title)
    formData.append('artist', artist)

    const res = await fetch('/api/generate', { method: 'POST', body: formData })
    const { task_id } = await res.json()
    setTaskId(task_id)

    const evtSource = new EventSource(`/api/progress/${task_id}`)
    evtSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setProgress(data.progress)
      setStatusMsg(data.message)

      if (data.status === 'completed') {
        setResultUrl(`/api/result/${task_id}`)
        setIsGenerating(false)
        evtSource.close()
      } else if (data.status === 'failed') {
        setStatusMsg(`Error: ${data.error}`)
        setIsGenerating(false)
        evtSource.close()
      }
    }
  }

  function handleNewVideo() {
    setStep(1)
    setFiles({})
    setConfig({ aspect: '9:16', theme: 'neon', lyricsStyle: 'karaoke', mode: 'full' })
    setTitle(localStorage.getItem('mv_title') || '')
    setArtist(localStorage.getItem('mv_artist') || '')
    setTaskId(null)
    setProgress(0)
    setStatusMsg('')
    setIsGenerating(false)
    setResultUrl(null)
  }

  const stepLabels = ['Upload', 'Theme', 'Lyrics', 'Info', 'Generate']

  return (
    <div className="app">
      <h1>MV Generate</h1>

      {/* Step indicator */}
      <div className="step-indicator">
        {stepLabels.map((label, i) => (
          <div key={i} className={`step-dot ${i + 1 === step ? 'active' : ''} ${i + 1 < step ? 'done' : ''}`}>
            <div className="dot">{i + 1 < step ? '✓' : i + 1}</div>
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Step 1: Upload Files */}
      {step === 1 && (
        <div className="wizard-step">
          <h2 className="step-title">Upload your files</h2>
          <p className="step-desc">Upload an MP3 audio file, lyrics text file, and a cover image.</p>
          <UploadArea files={files} onFilesChange={setFiles} />
        </div>
      )}

      {/* Step 2: Choose Theme */}
      {step === 2 && (
        <div className="wizard-step">
          <h2 className="step-title">Choose a visual theme</h2>
          <p className="step-desc">Select the visual style for your music video.</p>
          <ConfigPanel
            mode="theme"
            config={config}
            onConfigChange={setConfig}
          />
        </div>
      )}

      {/* Step 3: Choose Lyrics Style */}
      {step === 3 && (
        <div className="wizard-step">
          <h2 className="step-title">Choose lyrics display style</h2>
          <p className="step-desc">Select how lyrics appear in the video.</p>
          <ConfigPanel
            mode="lyrics"
            config={config}
            onConfigChange={setConfig}
          />
        </div>
      )}

      {/* Step 4: Song Info + Aspect Ratio */}
      {step === 4 && (
        <div className="wizard-step">
          <h2 className="step-title">Song details</h2>
          <p className="step-desc">Add song info and choose the video format.</p>

          <div className="info-form">
            <div className="form-group">
              <label>Song Title</label>
              <input
                placeholder="Enter song title..."
                value={title}
                onChange={(e) => { setTitle(e.target.value); localStorage.setItem('mv_title', e.target.value) }}
              />
            </div>
            <div className="form-group">
              <label>Artist</label>
              <input
                placeholder="Enter artist name..."
                value={artist}
                onChange={(e) => { setArtist(e.target.value); localStorage.setItem('mv_artist', e.target.value) }}
              />
            </div>
            <div className="form-group">
              <label>Video Mode</label>
              <div className="aspect-cards">
                {[
                  { value: 'full', label: 'Full Song', desc: 'Generate entire song' },
                  { value: 'chorus', label: 'Chorus Only', desc: 'Auto-detect & extract chorus' },
                ].map((opt) => (
                  <div
                    key={opt.value}
                    className={`aspect-card ${config.mode === opt.value ? 'selected' : ''}`}
                    onClick={() => setConfig({ ...config, mode: opt.value })}
                  >
                    <strong>{opt.label}</strong>
                    <span>{opt.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label>Aspect Ratio</label>
              <div className="aspect-cards">
                {[
                  { value: '9:16', label: '9:16', desc: 'TikTok / Reels' },
                  { value: '16:9', label: '16:9', desc: 'YouTube / Desktop' },
                ].map((opt) => (
                  <div
                    key={opt.value}
                    className={`aspect-card ${config.aspect === opt.value ? 'selected' : ''}`}
                    onClick={() => setConfig({ ...config, aspect: opt.value })}
                  >
                    <div className={`aspect-box ${opt.value === '9:16' ? 'portrait' : 'landscape'}`} />
                    <strong>{opt.label}</strong>
                    <span>{opt.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step 5: Generating / Result */}
      {step === 5 && (
        <div className="wizard-step">
          {isGenerating ? (
            <>
              <h2 className="step-title">Generating your video...</h2>
              <p className="step-desc">This may take a few minutes. Please wait.</p>
              <ProgressBar progress={progress} message={statusMsg} />
            </>
          ) : resultUrl ? (
            <>
              <h2 className="step-title">Your video is ready!</h2>
              <ResultView videoUrl={resultUrl} taskId={taskId!} />
              <button className="nav-btn secondary" onClick={handleNewVideo} style={{ marginTop: 16 }}>
                Create Another Video
              </button>
            </>
          ) : null}
        </div>
      )}

      {/* Navigation buttons */}
      {step < 5 && (
        <div className="wizard-nav">
          {step > 1 && (
            <button className="nav-btn secondary" onClick={() => setStep(step - 1)}>
              Back
            </button>
          )}
          <div style={{ flex: 1 }} />
          {step < 4 ? (
            <button
              className="nav-btn primary"
              disabled={!canGoNext()}
              onClick={() => setStep(step + 1)}
            >
              Next
            </button>
          ) : (
            <button
              className="nav-btn primary generate"
              onClick={handleGenerate}
            >
              Generate Video
            </button>
          )}
        </div>
      )}
    </div>
  )
}
