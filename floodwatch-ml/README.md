Flood Watch ML
# cross-platform (Python)
python run.py

# If using PowerShell and execution policy prevents running scripts:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

1st: Install dependencies
pip install -r requirements.txt

2nd: train model
python src/train_model.py

3rd: run prediction api
uvicorn src.serve_api:app --reload --host 127.0.0.1 --port 8000

sometimes error at src.util remove src then restart, paste back

open api to browser
http://127.0.0.1:8000/docs
or
http://localhost:8000/docs

test api
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Barangay":"Paliparan", "Elevation_m":84.5, "Duration_hr":3, "Rainfall_mm":22.6}'

Troubleshooting
1. “utils not found”
Run uvicorn from project root:

uvicorn src.serve_api:app --reload

2. Cannot activate venv
Run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

3. 0.0.0.0 not opening
Use:
http://127.0.0.1:8000/docs

4. Model not found
Train first:
python src/train_model.py

Barngay list api
http://127.0.0.1:8000/api/barangays

When running your backend (uvicorn src.serve_api:app --reload), the file will be available at
http://127.0.0.1:8000/static/data/FloodWatch_MLDataset.csv
