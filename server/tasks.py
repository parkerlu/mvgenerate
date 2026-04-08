import uuid
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result_path: str = ""
    error: str = ""


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = TaskInfo(task_id=task_id)
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for key, value in kwargs.items():
                    setattr(task, key, value)

    def run_in_background(self, task_id: str, fn: Callable, *args, **kwargs) -> None:
        def wrapper():
            self.update_task(task_id, status=TaskStatus.RUNNING)
            try:
                fn(*args, **kwargs)
                self.update_task(task_id, status=TaskStatus.COMPLETED, progress=1.0, message="Done")
            except Exception as e:
                self.update_task(task_id, status=TaskStatus.FAILED, error=str(e))

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()


task_manager = TaskManager()
