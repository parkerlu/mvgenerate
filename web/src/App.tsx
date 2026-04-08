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
}

export default function App() {
  const [files, setFiles] = useState<UploadedFiles>({})
  const [config, setConfig] = useState<Config>({
    aspect: '9:16',
    theme: 'neon',
    lyricsStyle: 'karaoke',
  })
  const [title, setTitle] = useState('')
  const [artist, setArtist] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)

  const canGenerate = files.audio && files.lyrics && files.cover && !isGenerating

  async function handleGenerate() {
    if (!files.audio || !files.lyrics || !files.cover) return

    setIsGenerating(true)
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

  return (
    <div className="app">
      <h1>MV Generate</h1>

      <div className="main-grid">
        <UploadArea files={files} onFilesChange={setFiles} />
        <ConfigPanel config={config} onConfigChange={setConfig} />
      </div>

      <div className="text-inputs section">
        <input
          placeholder="Song title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <input
          placeholder="Artist name"
          value={artist}
          onChange={(e) => setArtist(e.target.value)}
        />
      </div>

      <button
        className="generate-btn"
        disabled={!canGenerate}
        onClick={handleGenerate}
      >
        Generate Video
      </button>

      {isGenerating && (
        <ProgressBar progress={progress} message={statusMsg} />
      )}

      {resultUrl && (
        <ResultView videoUrl={resultUrl} taskId={taskId!} />
      )}
    </div>
  )
}
