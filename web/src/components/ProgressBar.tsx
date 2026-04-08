interface Props {
  progress: number
  message: string
}

export default function ProgressBar({ progress, message }: Props) {
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
    </div>
  )
}
