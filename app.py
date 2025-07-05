# === Project: Video Subtitle Processor ===
# Clean Architecture Refactored Version with API Router Separation

# ───────────────────────────────────────────────────────────
# Structure:
# - app.py (entry point)
# - api/
#     └── routes.py            → All FastAPI route handlers
# - services/
#     └── processor.py         → Processing logic
# - utils/
#     └── helpers.py           → Utility functions
#     └── pdf_generator.py     → PDF creation class
# - db/
#     └── mongo.py             → MongoDB client & collections
# - templates/
#     └── upload_video.html
# - uploads/, pdfs/, DejaVuSans.ttf
# ───────────────────────────────────────────────────────────

# app.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from api.routes import router as api_router

UPLOAD_FOLDER = "uploads"
PDF_FOLDER = "pdfs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.include_router(api_router)

# templates directory
templates = Jinja2Templates(directory="templates")