import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app

# Vercel will execute this module and use the Flask `app` object.
# The app is imported from the existing project entry point so the UI,
# routes, prediction logic, and model behavior remain unchanged.
