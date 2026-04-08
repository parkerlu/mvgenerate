#!/usr/bin/env python3
"""One-command launcher: starts FastAPI backend and serves the frontend."""
import subprocess
import sys
import os
from pathlib import Path


def main():
    # Build frontend if dist doesn't exist
    web_dir = Path("web")
    dist_dir = web_dir / "dist"

    if not dist_dir.exists():
        print("Building frontend...")
        if not (web_dir / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=str(web_dir), check=True)
        subprocess.run(["npx", "vite", "build"], cwd=str(web_dir), check=True)

    # Start FastAPI server
    print("Starting MV Generate server at http://localhost:8000")
    os.execvp(sys.executable, [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ])


if __name__ == "__main__":
    main()
