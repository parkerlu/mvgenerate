import { useRef, useState } from 'react'

interface UploadedFile {
  path: string
  name: string
  previewUrl?: string
}

interface Props {
  files: {
    audio?: UploadedFile
    lyrics?: UploadedFile
    cover?: UploadedFile
  }
  onFilesChange: (files: Props['files']) => void
}

export default function UploadArea({ files, onFilesChange }: Props) {
  const audioRef = useRef<HTMLInputElement>(null)
  const coverRef = useRef<HTMLInputElement>(null)
  const lyricsRef = useRef<HTMLInputElement>(null)
  const [lyricsText, setLyricsText] = useState('')
  const [lyricsMode, setLyricsMode] = useState<'text' | 'file'>('text')

  async function handleUpload(
    file: File,
    key: 'audio' | 'lyrics' | 'cover',
  ) {
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch('/api/upload', { method: 'POST', body: formData })
    const data = await res.json()

    const uploaded: UploadedFile = { path: data.path, name: file.name }

    if (key === 'cover') {
      uploaded.previewUrl = URL.createObjectURL(file)
    }

    onFilesChange({ ...files, [key]: uploaded })
  }

  async function handleLyricsText(text: string) {
    setLyricsText(text)
    if (!text.trim()) {
      onFilesChange({ ...files, lyrics: undefined })
      return
    }
    // Upload text as a file via blob
    const blob = new Blob([text], { type: 'text/plain' })
    const file = new File([blob], 'lyrics.txt', { type: 'text/plain' })
    const formData = new FormData()
    formData.append('file', file)
    const res = await fetch('/api/upload', { method: 'POST', body: formData })
    const data = await res.json()
    onFilesChange({ ...files, lyrics: { path: data.path, name: 'lyrics.txt' } })
  }

  // Debounce lyrics text upload
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  function onLyricsChange(text: string) {
    setLyricsText(text)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => handleLyricsText(text), 500)
  }

  function renderZone(
    key: 'audio' | 'cover',
    label: string,
    accept: string,
    ref: React.RefObject<HTMLInputElement | null>,
  ) {
    const file = files[key]
    return (
      <div
        className={`upload-zone ${file ? 'uploaded' : ''}`}
        onClick={() => ref.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          const f = e.dataTransfer.files[0]
          if (f) handleUpload(f, key)
        }}
      >
        <input
          ref={ref}
          type="file"
          accept={accept}
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) handleUpload(f, key)
          }}
        />
        <p>{label}</p>
        {file && <p className="filename">{file.name}</p>}
        {key === 'cover' && file?.previewUrl && (
          <img src={file.previewUrl} alt="cover" className="cover-preview" />
        )}
      </div>
    )
  }

  return (
    <div className="section">
      <h2>Upload Files</h2>
      {renderZone('audio', 'Drop MP3 file here', '.mp3,audio/*', audioRef)}

      {/* Lyrics: toggle between text input and file upload */}
      <div className="lyrics-section">
        <div className="lyrics-tabs">
          <button
            className={`lyrics-tab ${lyricsMode === 'text' ? 'active' : ''}`}
            onClick={() => setLyricsMode('text')}
          >
            Paste Lyrics
          </button>
          <button
            className={`lyrics-tab ${lyricsMode === 'file' ? 'active' : ''}`}
            onClick={() => setLyricsMode('file')}
          >
            Upload File
          </button>
        </div>

        {lyricsMode === 'text' ? (
          <textarea
            className="lyrics-textarea"
            placeholder={"Paste lyrics here...\nSupports Suno format ([Verse], [Chorus] etc.)"}
            value={lyricsText}
            onChange={(e) => onLyricsChange(e.target.value)}
            rows={8}
          />
        ) : (
          <div
            className={`upload-zone ${files.lyrics ? 'uploaded' : ''}`}
            onClick={() => lyricsRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              const f = e.dataTransfer.files[0]
              if (f) handleUpload(f, 'lyrics')
            }}
          >
            <input
              ref={lyricsRef}
              type="file"
              accept=".txt,text/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleUpload(f, 'lyrics')
              }}
            />
            <p>Drop lyrics .txt file here</p>
            {files.lyrics && <p className="filename">{files.lyrics.name}</p>}
          </div>
        )}
        {files.lyrics && lyricsMode === 'text' && (
          <p className="filename" style={{ textAlign: 'right', marginTop: 4 }}>Lyrics saved</p>
        )}
      </div>

      {renderZone('cover', 'Drop cover image here', '.jpg,.jpeg,.png,image/*', coverRef)}
    </div>
  )
}
