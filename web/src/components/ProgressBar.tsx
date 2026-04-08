import { useState, useEffect } from 'react'

interface Props {
  progress: number
  message: string
}

export default function ProgressBar({ progress, message }: Props) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const start = Date.now()
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000))
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const mins = Math.floor(elapsed / 60)
  const secs = elapsed % 60
  const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return (
    <div className="section progress-section">
      <div className="progress-bar-bg">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.round(progress * 100)}%` }}
        />
      </div>
      <p className="progress-text">
        {Math.round(progress * 100)}% — {message}
      </p>
      <p className="progress-text" style={{ marginTop: 4, fontSize: '0.8rem' }}>
        Elapsed: {timeStr}
      </p>
    </div>
  )
}
