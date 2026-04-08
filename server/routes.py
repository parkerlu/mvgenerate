import json
import shutil
import asyncio
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from config import AspectRatio, Theme, LyricsStyle, GenerateMode, GenerateConfig
from server.tasks import task_manager, TaskStatus
from mvgenerate import generate

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an MP3, lyrics txt, or cover image. Returns a file ID."""
    file_id = task_manager.create_task()[:8]
    ext = Path(file.filename).suffix if file.filename else ""
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "filename": file.filename, "path": str(save_path)}


@router.post("/generate")
async def generate_video(
    audio_path: str = Form(...),
    lyrics_path: str = Form(...),
    cover_path: str = Form(...),
    aspect: str = Form("9:16"),
    theme: str = Form("neon"),
    lyrics_style: str = Form("karaoke"),
    mode: str = Form("full"),
    title: str = Form(""),
    artist: str = Form(""),
):
    """Start video generation. Returns a task ID for progress tracking."""
    for p, name in [(audio_path, "audio"), (lyrics_path, "lyrics"), (cover_path, "cover")]:
        if not Path(p).exists():
            raise HTTPException(400, f"{name} file not found: {p}")

    task_id = task_manager.create_task()
    output_path = OUTPUT_DIR / f"{task_id}.mp4"

    config = GenerateConfig(
        audio_path=Path(audio_path),
        lyrics_path=Path(lyrics_path),
        cover_path=Path(cover_path),
        output_path=output_path,
        aspect=AspectRatio(aspect),
        theme=Theme(theme),
        lyrics_style=LyricsStyle(lyrics_style),
        mode=GenerateMode(mode),
        title=title,
        artist=artist,
    )

    def progress_callback(msg: str, pct: float):
        task_manager.update_task(task_id, message=msg, progress=pct)

    def run():
        generate(config, progress_callback=progress_callback)
        task_manager.update_task(task_id, result_path=str(output_path))

    task_manager.run_in_background(task_id, run)
    return {"task_id": task_id}


@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """SSE endpoint for real-time progress updates."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    async def event_stream():
        while True:
            task = task_manager.get_task(task_id)
            if task is None:
                break

            data = {
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
            }

            if task.status == TaskStatus.COMPLETED:
                data["result_path"] = task.result_path
                yield f"data: {json.dumps(data)}\n\n"
                break
            elif task.status == TaskStatus.FAILED:
                data["error"] = task.error
                yield f"data: {json.dumps(data)}\n\n"
                break

            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    """Download the generated video."""
    # Try task manager first
    task = task_manager.get_task(task_id)
    if task and task.status == TaskStatus.COMPLETED and Path(task.result_path).exists():
        return FileResponse(
            task.result_path,
            media_type="video/mp4",
            filename=f"mv_{task_id}.mp4",
        )

    # Fallback: check if file exists on disk (e.g. after server restart)
    fallback_path = OUTPUT_DIR / f"{task_id}.mp4"
    if fallback_path.exists() and fallback_path.stat().st_size > 0:
        return FileResponse(
            str(fallback_path),
            media_type="video/mp4",
            filename=f"mv_{task_id}.mp4",
        )

    if task and task.status == TaskStatus.FAILED:
        raise HTTPException(400, f"Generation failed: {task.error}")
    if task and task.status == TaskStatus.RUNNING:
        raise HTTPException(400, "Still generating, please wait")

    raise HTTPException(404, "Result not found")
    )
