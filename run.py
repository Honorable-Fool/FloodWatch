# run.py — Cross-platform launcher for backend + frontend

import os
import subprocess
import time
import webbrowser
from pathlib import Path
import venv

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"

# Backend
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_APP = "src.serve_api:app"
BACKEND_CWD = ROOT / "floodwatch-ml"

# Frontend
FRONTEND_DIR = ROOT
FRONTEND_PORT = 5500


def ensure_venv():
    if not VENV_DIR.exists():
        print("Creating .venv...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(".venv already exists.")


def pip_install():
    python_exe = (
        VENV_DIR / "Scripts" / "python.exe"
        if os.name == "nt"
        else VENV_DIR / "bin" / "python"
    )
    # prefer invoking pip via `python -m pip` — this works reliably on Windows
    # (calling pip.exe directly can fail when pip self-upgrades)
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

    if REQ_FILE.exists():
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(REQ_FILE)])
    else:
        print("No requirements.txt found — skipping.")

    # quick sanity check for uvicorn — if not installed, warn user (backend won't start)
    try:
        check_uvicorn = subprocess.run([str(python_exe), "-c", "import uvicorn"], capture_output=True)
        if check_uvicorn.returncode != 0:
            print("Warning: 'uvicorn' not found in the venv. Backend won't start until it's installed.")
            print(f"Install with: {python_exe} -m pip install uvicorn[standard]")
    except Exception:
        # best-effort; don't fail the launcher if the check fails
        pass

    return python_exe


def start_backend(python_exe):
    print("Starting FastAPI backend...")
    return subprocess.Popen(
        [
            str(python_exe),
            "-m",
            "uvicorn",
            BACKEND_APP,
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=str(BACKEND_CWD),
    )


def start_frontend(python_exe):
    print("Starting frontend static server...")
    return subprocess.Popen(
        [str(python_exe), "-m", "http.server", str(FRONTEND_PORT)],
        cwd=str(FRONTEND_DIR),
    )


def main():
    ensure_venv()
    python_exe = pip_install()

    backend_proc = start_backend(python_exe)
    time.sleep(1)

    frontend_proc = start_frontend(python_exe)
    time.sleep(1)

    url = f"http://127.0.0.1:{FRONTEND_PORT}/index.html"
    print(f"Opening browser at {url}")
    webbrowser.open(url)

    print("FloodWatch system running.")
    print("Press Ctrl+C to stop.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("Stopping processes...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
