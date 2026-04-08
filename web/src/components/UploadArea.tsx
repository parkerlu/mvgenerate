import { useRef } from 'react'

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
  const lyricsRef = useRef<HTMLInputElement>(null)
  const coverRef = useRef<HTMLInputElement>(null)

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

  function renderZone(
    key: 'audio' | 'lyrics' | 'cover',
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
      {renderZone('lyrics', 'Drop lyrics .txt file here', '.txt,text/*', lyricsRef)}
      {renderZone('cover', 'Drop cover image here', '.jpg,.jpeg,.png,image/*', coverRef)}
    </div>
  )
}
