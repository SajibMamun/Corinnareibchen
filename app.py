import os
import uuid
import time
import threading
import ffmpeg
import whisper
from datetime import datetime
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fpdf import FPDF
from io import BytesIO

# === CONFIGURATION ===
UPLOAD_FOLDER = "uploads"
PDF_FOLDER = "pdfs"
FONT_FILE = "DejaVuSans.ttf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)



templates = Jinja2Templates(directory="templates")




# Load Whisper model once
whisper_model = whisper.load_model("medium")  # or "medium"





# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["Corinnareibchen"]
video_collection = db["video"]
subtitles_collection = db["subtitles"]





# PDF class
class UnicodePDF(FPDF):
    def __init__(self, video_id):
        super().__init__()
        self.video_id = video_id

    def header(self):
        self.set_font("DejaVu", size=12)
        self.cell(0, 10, "Subtitles Extracted from Video", ln=True, align="C")
        self.ln(5)
        self.set_font("DejaVu", size=10)
        self.cell(0, 10, f"Video ID: {self.video_id}", ln=True, align="C")
        self.ln(5)

# Helpers
def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ar='16000').run(overwrite_output=True)

def transcribe_whisper(audio_path):
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.get("segments", [])]

def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

def clean_uploads(delay=30):
    time.sleep(delay)
    for file in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error deleting {path}: {e}")

@app.get("/", response_class=HTMLResponse)
def serve_upload_form(request: Request):
    return templates.TemplateResponse("upload_video.html", {"request": request})





@app.post("/upload_video/")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file uploaded")

    video_id = str(uuid.uuid4())[:8]
    video_filename = f"{video_id}_{video.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)

    try:
        with open(video_path, "wb") as f:
            f.write(await video.read())

        doc = {
            "video_id": video_id,
            "video_filename": video_filename,
            "video_path": video_path,
            "uploaded_at": datetime.utcnow()
        }
        print("Inserting into DB:", doc)
        video_collection.insert_one(doc)
        print("Inserted successfully")
    except Exception as e:
        print("Upload or DB insert failed:", e)
        raise HTTPException(status_code=500, detail="Failed to upload or save to database")

    return {
        "message": "Video uploaded successfully",
        "video_id": video_id,
        "next_step": f"/process_from_file/{video_id}"
    }







def process_video_from_file(video_id: str):
    video_doc = video_collection.find_one({"video_id": video_id})
    if not video_doc:
        raise Exception("Video not found")

    video_path = video_doc["video_path"]
    video_filename = video_doc["video_filename"]
    if not os.path.exists(video_path):
        raise Exception("Video file missing on disk")

    audio_path = os.path.splitext(video_path)[0] + ".wav"
    extract_audio(video_path, audio_path)
    segments = transcribe_whisper(audio_path)
    os.remove(audio_path)

    pdf = UnicodePDF(video_id)
    pdf.add_font("DejaVu", "", FONT_FILE, uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()

    for seg in segments:
        line = f"[{format_time(seg['start'])} - {format_time(seg['end'])}] {seg['text'].strip()}"
        try:
            pdf.multi_cell(w=0, h=10, txt=line)
            pdf.ln(2)
        except Exception as e:
            print("PDF write error:", e)
            pdf.multi_cell(w=0, h=10, txt="[Skipped line due to error]")

    pdf_filename = f"{os.path.splitext(video_filename)[0]}_subtitles_{video_id}.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    pdf.output(pdf_path)

    subtitles_collection.insert_one({
        "video_id": video_id,
        "video_filename": video_filename,
        "pdf_filename": pdf_filename,
        "transcript": segments,
        "created_at": datetime.utcnow()
    })

    return pdf_path, pdf_filename







@app.post("/process_from_file/{video_id}")
def process_from_file(video_id: str):
    try:
        pdf_path, pdf_filename = process_video_from_file(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    threading.Thread(target=clean_uploads, daemon=True).start()
    return FileResponse(pdf_path, filename=pdf_filename, media_type="application/pdf")







@app.get("/pdf/{video_id}")
def download_pdf(video_id: str):
    doc = subtitles_collection.find_one({"video_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_path = os.path.join(PDF_FOLDER, doc["pdf_filename"])
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not on server")

    return FileResponse(pdf_path, media_type="application/pdf", filename=doc["pdf_filename"])







@app.get("/transcript/{video_id}")
def get_transcript(video_id: str):
    doc = subtitles_collection.find_one({"video_id": video_id}, {"_id": 0, "transcript": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return JSONResponse(content=doc)







@app.delete("/delete/{video_id}")
def delete_video_data(video_id: str):
    video_doc = video_collection.find_one({"video_id": video_id})
    if video_doc:
        video_path = video_doc.get("video_path")
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        video_collection.delete_one({"video_id": video_id})

    subtitle_doc = subtitles_collection.find_one({"video_id": video_id})
    if subtitle_doc:
        pdf_filename = subtitle_doc.get("pdf_filename")
        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        subtitles_collection.delete_one({"video_id": video_id})

    return {"message": "Deleted successfully", "video_id": video_id}






@app.get("/print_collections")
def print_collections():
    videos = list(video_collection.find({}, {"_id": 0}))
    subtitles = list(subtitles_collection.find({}, {"_id": 0}))
    return {"videos": videos, "subtitles": subtitles}