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

# ================= ML Backend Configuration =================
ML_PROJECT_ROOT = ROOT / "floodwatch-ml"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_APP = "src.serve_api:app"
BACKEND_CWD = ML_PROJECT_ROOT 

# ML Model Paths
ML_MODEL_FILE = ML_PROJECT_ROOT / "models" / "random_forest_floodwatch.joblib"
ML_TRAIN_SCRIPT = ML_PROJECT_ROOT / "src" / "train_model.py"

# ================= NLP Backend Configuration =================
NLP_PROJECT_ROOT = ROOT / "floodwatch-nlp"
NLP_DIR = NLP_PROJECT_ROOT / "src"  # We will run the app from here
NLP_APP = "app.py"
NLP_PORT = 8001
NLP_MODEL_FILE = NLP_PROJECT_ROOT / "models_nlp" / "nb_model.pkl"
NLP_TRAIN_SCRIPT = NLP_PROJECT_ROOT / "src" / "train_nlp.py"

# ================= Frontend Configuration =================
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
    
    # Check pip
    check_pip = subprocess.run([str(python_exe), "-m", "pip", "--version"], capture_output=True)
    if check_pip.returncode != 0:
        print("pip not found in venv — attempting to bootstrap pip...")
        try:
            subprocess.check_call([str(python_exe), "-m", "ensurepip", "--upgrade"])
            print("Bootstrapped pip with ensurepip.")
        except subprocess.CalledProcessError:
            try:
                import tempfile
                import urllib.request
                print("ensurepip failed — downloading get-pip.py...")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix="-get-pip.py")
                urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", tmp.name)
                tmp.close()
                subprocess.check_call([str(python_exe), tmp.name])
                os.unlink(tmp.name)
                print("Bootstrapped pip with get-pip.py.")
            except Exception:
                print("Failed to bootstrap pip — continuing.")

    # Upgrade pip
    try:
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("Warning: failed to upgrade pip.")

    # Install requirements
    if REQ_FILE.exists():
        try:
            subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(REQ_FILE)])
        except subprocess.CalledProcessError:
            print("Warning: installing requirements failed.")
    else:
        print("No requirements.txt found — skipping.")

    # Check uvicorn
    try:
        check_uvicorn = subprocess.run([str(python_exe), "-c", "import uvicorn"], capture_output=True)
        if check_uvicorn.returncode != 0:
            print("Warning: 'uvicorn' not found. Backend might not start.")
    except Exception:
        pass

    return python_exe


# ================= ML HELPER FUNCTIONS (NEW) =================

def check_ml_model():
    """Check if ML model exists"""
    if ML_MODEL_FILE.exists():
        print(f"✓ ML model found: {ML_MODEL_FILE.name}")
        return True
    else:
        print(f"⚠ ML model not found at: {ML_MODEL_FILE}")
        return False

def train_ml_model(python_exe):
    """Train the ML model"""
    print("\n" + "="*50)
    print("TRAINING ML MODEL")
    print("="*50)
    print("This may take a few minutes...")
    
    try:
        # Run inside floodwatch-ml root so imports work correctly
        subprocess.check_call(
            [str(python_exe), str(ML_TRAIN_SCRIPT)],
            cwd=str(ML_PROJECT_ROOT) 
        )
        print("✓ ML model training completed!")
        return True
    except subprocess.CalledProcessError:
        print("✗ ML model training failed. Check logs.")
        return False
    except FileNotFoundError:
        print(f"✗ Training script not found: {ML_TRAIN_SCRIPT}")
        return False

def start_ml_backend(python_exe):
    """Start the ML FastAPI backend"""
    print("Starting ML Backend (FastAPI)...")
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


# ================= NLP HELPER FUNCTIONS =================

def check_nlp_model():
    """Check if NLP model exists"""
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
        subprocess.check_call(
            [str(python_exe), str(NLP_TRAIN_SCRIPT)],
            cwd=str(NLP_DIR) 
        )
        print("✓ NLP model training completed!")
        return True
    except subprocess.CalledProcessError:
        print("✗ NLP model training failed.")
        return False
    except FileNotFoundError:
        print(f"✗ Training script not found: {NLP_TRAIN_SCRIPT}")
        return False

def start_nlp_backend(python_exe):
    """Start NLP backend API server"""
    print("Starting NLP Backend API...")
    nlp_app_path = NLP_DIR / NLP_APP
    
    if not nlp_app_path.exists():
        print(f"⚠ NLP backend not found: {nlp_app_path}")
        return None
    
    try:
        return subprocess.Popen(
            [str(python_exe), str(NLP_APP)],
            cwd=str(NLP_DIR),
        )
    except Exception as e:
        print(f"✗ Failed to start NLP backend: {e}")
        return None


def start_frontend(python_exe):
    print("Starting Frontend (Static Server)...")
    return subprocess.Popen(
        [str(python_exe), "-m", "http.server", str(FRONTEND_PORT)],
        cwd=str(FRONTEND_DIR),
    )


def main():
    # 1. Setup Virtual Environment and Install Dependencies
    ensure_venv()
    python_exe = pip_install()

    # 2. NLP Backend Setup (Check -> Train -> Start)
    nlp_proc = None
    if NLP_DIR.exists():
        print("\n" + "="*50)
        print("Checking NLP Backend...")
        print("="*50)
        
        if not check_nlp_model():
            response = input("Train NLP model now? (y/n): ").lower().strip()
            if response == 'y':
                if train_nlp_model(python_exe):
                    nlp_proc = start_nlp_backend(python_exe)
                else:
                    print("⚠ Continuing without NLP backend")
            else:
                print("⚠ Skipping NLP backend (model not trained)")
        else:
            nlp_proc = start_nlp_backend(python_exe)
        
        time.sleep(1)
    else:
        print("⚠ NLP directory not found, skipping NLP backend")

    # 3. ML Backend Setup (Check -> Train -> Start)
    if ML_PROJECT_ROOT.exists():
        print("\n" + "="*50)
        print("Checking ML Backend...")
        print("="*50)
        
        if not check_ml_model():
            response = input("Train ML model now? (y/n): ").lower().strip()
            if response == 'y':
                train_ml_model(python_exe)
                # We assume training saves the file where the backend expects it
            else:
                print("⚠ Skipping ML model training (Backend will run in fallback mode)")
        else:
            # Model exists, nothing to do
            pass
    else:
        print("⚠ ML directory not found.")

    # Start ML Backend (it starts even if training failed, just warns)
    ml_proc = start_ml_backend(python_exe)
    time.sleep(1)

    # 4. Start Frontend
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
    else:
        print(f"NLP API: Not running (Check logs)")

    if ml_proc and ml_proc.poll() is None:
        print(f"ML API:  http://{BACKEND_HOST}:{BACKEND_PORT}")
    else:
        print(f"ML API:  Not running (Check logs)")

    print(f"Frontend: http://127.0.0.1:{FRONTEND_PORT}")
    print("="*50)
    print("Press Ctrl+C to stop.")

    try:
        ml_proc.wait()
        frontend_proc.wait()
        if nlp_proc:
            nlp_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping processes...")
        ml_proc.terminate()
        frontend_proc.terminate()
        if nlp_proc:
            nlp_proc.terminate()


if __name__ == "__main__":
    main()