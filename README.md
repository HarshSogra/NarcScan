# NarcScan

NarcScan is a narcotics field-test analysis tool being built as an SIH 2026 prototype.
It uses a Python/FastAPI backend and a Flutter mobile app.

---

## Phase 1 — Backend Setup

Phase 1 sets up the FastAPI server with a single health-check endpoint.
No AI, no database, no camera features yet — just a working server.

---

## How to run (Windows)

**1. Create the virtual environment**
```
cd backend
python -m venv .venv
```

**2. Activate the virtual environment**
```
.venv\Scripts\activate
```

**3. Install dependencies**
```
pip install -r requirements.txt
```

**4. Start the FastAPI server**
```
uvicorn app.main:app --reload
```

**5. Run the health test**
```
pytest tests/test_health.py -v
```
