#!/bin/bash
# Start the Knowledge Base API server

cd "$(dirname "$0")"

# Install dependencies (if needed)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

# Launch
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
