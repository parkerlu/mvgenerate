interface Props {
  videoUrl: string
  taskId: string
}

export default function ResultView({ videoUrl, taskId }: Props) {
  return (
    <div className="section result-section">
      <h2>Result</h2>
      <video src={videoUrl} controls />
      <div>
        <a className="download-btn" href={videoUrl} download={`mv_${taskId}.mp4`}>
          Download MP4
        </a>
      </div>
    </div>
  )
}
