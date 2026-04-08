# CardioAgentAI
A cardiovascular risk assessment web app using the Framingham Risk Score model. Built with a Python Flask backend and a modern  frontend.# CardioCheck — Grinova 2025



---

## Project Structure

```
CardioCheck-Grinova-2025/
├── backend/        # Flask API — risk scoring logic
│   ├── app.py
│   └── venv/
├── frontend/       # Node.js UI
│   └── ...
└── README.md
```

---

## Prerequisites

- Python 3.8+
- Node.js (via NVM recommended)
- `venv` module available

---

## Setup & Run

### 1. Backend

```bash
cd backend

# First time only — create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate         # Linux/macOS
# OR
venv\Scripts\activate            # Windows

# First time only — install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

> Backend runs on **http://localhost:5000**

---

### 2. Frontend

```bash
cd frontend

# Load NVM (if using NVM)
export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh"

# First time only
npm install

# Start dev server
npm run dev
```

> Frontend runs on **http://localhost:5173** (or as configured)

---

## How It Works

The app calculates a **10-year cardiovascular risk percentage** using the Framingham Risk Score, which takes into account:

- Age & Sex
- Total Cholesterol & HDL
- Systolic Blood Pressure
- Smoking status
- Blood pressure treatment status

The Flask backend processes these inputs and returns the risk score. The frontend provides a clean form UI to submit data and display results.

---

## Quick Reference

| Command | What it does |
|---|---|
| `source venv/bin/activate` | Activate Python env |
| `python app.py` | Start Flask backend |
| `npm run dev` | Start frontend dev server |
| `deactivate` | Exit Python venv |

---

## License

MIT © 2025 Shido/Voiid
