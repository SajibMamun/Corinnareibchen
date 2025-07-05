import os
from datetime import datetime
from bson import ObjectId
from db.mongo import video_collection, subtitles_collection
from utils.helpers import extract_audio, transcribe_whisper, format_time
from utils.pdf_generator import UnicodePDF

PDF_FOLDER = "pdfs"
FONT_FILE = "DejaVuSans.ttf"

def process_video_from_file(object_id: str):
    try:
        video_doc = video_collection.find_one({"_id": ObjectId(object_id)})
    except Exception:
        raise Exception("Invalid ObjectId format")

    if not video_doc:
        raise Exception("Video not found")

    video_id = video_doc["video_id"]
    video_path = video_doc["video_path"]
    video_filename = video_doc["video_filename"]

    if not os.path.exists(video_path):
        raise Exception("Video file not found")

    audio_path = os.path.splitext(video_path)[0] + ".wav"
    extract_audio(video_path, audio_path)
    segments = transcribe_whisper(audio_path)
    os.remove(audio_path)

    pdf = UnicodePDF(video_id, object_id, video_filename)
    pdf.add_font("DejaVu", "", FONT_FILE, uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()

    for seg in segments:
        line = f"[{format_time(seg['start'])} - {format_time(seg['end'])}] {seg['text'].strip()}"
        pdf.multi_cell(0, 10, line)
        pdf.ln(2)

    pdf_filename = f"{os.path.splitext(video_filename)[0]}_subtitles_{video_id}.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    pdf.output(pdf_path)

    subtitles_collection.insert_one({
        "video_id": video_id,
        "video_object_id": ObjectId(object_id),
        "video_filename": video_filename,
        "pdf_filename": pdf_filename,
        "transcript": segments,
        "created_at": datetime.utcnow()
    })

    return pdf_path, pdf_filename