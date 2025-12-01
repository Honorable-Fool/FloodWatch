# run.py — Cross-platform launcher for backend + frontend

import os
import subprocess
import time
import webbrowser
from pathlib import Path
import venv
from app import app

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"

# Backend
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_APP = "src.serve_api:app"
# NOTE: This path assumes you have a subfolder called 'floodwatch-ml' 
# containing your FastAPI app. If your FastAPI app is also in the root 
# folder, you should change this to BACKEND_CWD = ROOT.
BACKEND_CWD = ROOT / "floodwatch-ml" 

# NLP Backend (FIXED PATHS)
NLP_DIR = ROOT 
NLP_APP = "app.py"
NLP_PORT = 5000
NLP_MODEL_FILE = ROOT / "models" / "nb_model.pkl" 
NLP_TRAIN_SCRIPT = ROOT / "train_nlp.py" 

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
    # Ensure the venv's Python can run pip. Some Python installs/venvs don't
    # have pip available out of the box — try a few fallbacks to bootstrap it.
    check_pip = subprocess.run([str(python_exe), "-m", "pip", "--version"], capture_output=True)
    if check_pip.returncode != 0:
        print("pip not found in venv — attempting to bootstrap pip...")
        # First try ensurepip (bundled in many Python builds)
        try:
            subprocess.check_call([str(python_exe), "-m", "ensurepip", "--upgrade"])
            print("Bootstrapped pip with ensurepip.")
        except subprocess.CalledProcessError:
            # As a last resort, download get-pip.py and run it (best-effort).
            try:
                import tempfile
                import urllib.request

                print("ensurepip failed — downloading get-pip.py to bootstrap pip...")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix="-get-pip.py")
                urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", tmp.name)
                tmp.close()
                subprocess.check_call([str(python_exe), tmp.name])
                os.unlink(tmp.name)
                print("Bootstrapped pip with get-pip.py.")
            except Exception:
                print("Failed to bootstrap pip in the venv — continuing but pip calls may fail.")

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


# ============== NLP FUNCTIONS (ADDED) ==============

def check_nlp_model():
    """Check if NLP model exists, offer to train if not"""
    if NLP_MODEL_FILE.exists():
        print(f"✓ NLP model found: {NLP_MODEL_FILE.name}")
        return True
    else:
        print(f"⚠ NLP model not found at: {NLP_MODEL_FILE}")
        return False


def train_nlp_model(python_exe):
    """Train the NLP model"""
    print("\n" + "="*50)
    print("TRAINING NLP MODEL")
    print("="*50)
    print("This may take a few minutes...")
    
    try:
        # NOTE: NLP_DIR is ROOT, which is the FloodWatch-main folder.
        # This is correct since the data file is in the root folder.
        subprocess.check_call(
            [str(python_exe), str(NLP_TRAIN_SCRIPT)],
            cwd=str(NLP_DIR) 
        )
        print("✓ NLP model training completed!")
        return True
    except subprocess.CalledProcessError:
        print("✗ NLP model training failed. Check train_nlp.py logs for details.")
        return False
    except FileNotFoundError:
        print(f"✗ Training script not found: {NLP_TRAIN_SCRIPT}")
        return False


def start_nlp_backend(python_exe):
    """Start NLP backend API server"""
    print("Starting NLP backend API...")
    nlp_app_path = NLP_DIR / NLP_APP
    
    if not nlp_app_path.exists():
        print(f"⚠ NLP backend not found: {nlp_app_path}")
        return None
    
    try:
        proc = subprocess.Popen(
            [str(python_exe), str(nlp_app_path)],
            cwd=str(NLP_DIR),
        )
        print(f"✓ NLP backend started at http://{BACKEND_HOST}:{NLP_PORT}")
        return proc
    except Exception as e:
        print(f"✗ Failed to start NLP backend: {e}")
        return None

# ===================================================


def start_backend(python_exe):
    print("Starting FastAPI backend...")
    # This command uses uvicorn to run the main backend
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
    # This command uses Python's simple http server to serve the frontend
    return subprocess.Popen(
        [str(python_exe), "-m", "http.server", str(FRONTEND_PORT)],
        cwd=str(FRONTEND_DIR),
    )


def main():
    # 1. Setup Virtual Environment and Install Dependencies
    ensure_venv()
    python_exe = pip_install()

    # 2. NLP Backend Setup (Model Training and API Start)
    nlp_proc = None
    if NLP_DIR.exists():
        print("\n" + "="*50)
        print("Checking NLP Backend...")
        print("="*50)
        
        if not check_nlp_model():
            response = input("Train NLP model now? (y/n): ").lower().strip()
            if response == 'y':
                if train_nlp_model(python_exe):
                    # Only start API if training succeeded
                    nlp_proc = start_nlp_backend(python_exe)
                else:
                    print("⚠ Continuing without NLP backend")
            else:
                print("⚠ Skipping NLP backend (model not trained)")
        else:
            # If model exists, just start the API
            nlp_proc = start_nlp_backend(python_exe)
        
        time.sleep(1)
    else:
        print("⚠ NLP directory not found, skipping NLP backend")

    # 3. Start Main Backend (FastAPI)
    backend_proc = start_backend(python_exe)
    time.sleep(1)

    # 4. Start Frontend (Static Server)
    frontend_proc = start_frontend(python_exe)
    time.sleep(1)

    # 5. Open Browser
    url = f"http://127.0.0.1:{FRONTEND_PORT}/index.html"
    print(f"Opening browser at {url}")
    webbrowser.open(url)

    # 6. Monitor Processes
    print("\n" + "="*50)
    print("FloodWatch system running.")
    print("="*50)
    if nlp_proc and nlp_proc.poll() is None:
        print(f"NLP API: http://{BACKEND_HOST}:{NLP_PORT}")
    print(f"Backend: http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"Frontend: http://127.0.0.1:{FRONTEND_PORT}")
    print("="*50)
    print("Press Ctrl+C to stop.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
        if nlp_proc:
            nlp_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping processes...")
        backend_proc.terminate()
        frontend_proc.terminate()
        if nlp_proc:
            nlp_proc.terminate()


if __name__ == "__main__":
    main()