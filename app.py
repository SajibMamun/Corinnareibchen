import os
import uuid
from datetime import datetime
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from bson import ObjectId
from pymongo import MongoClient, errors
import ffmpeg
import whisper
from fpdf import FPDF
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pydantic import BaseModel

# ───────────── Configuration ─────────────
UPLOAD_FOLDER = "uploads"
PDF_FOLDER = "pdfs"
FONT_FILE = "DejaVuSans.ttf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Load environment variables and configure API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ───────────── MongoDB Setup ─────────────
try:
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["crryptoEducation"]
    video_collection = db["video"]
    subtitles_collection = db["subtitles"]
    global_pdfs_collection = db["global_pdfs"]


except errors.ServerSelectionTimeoutError as err:
    raise Exception(f"Could not connect to MongoDB: {err}")

# ───────────── Whisper Model Setup ─────────────
whisper_model = whisper.load_model("base")

# ───────────── ChromaDB Setup ─────────────
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "video_pdf_knowledge"
GLOBAL_COLLECTION_NAME = "global_pdf_knowledge"


chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
global_knowledge_collection = chroma_client.get_or_create_collection(name=GLOBAL_COLLECTION_NAME)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")






# ───────────── Utility Functions ─────────────

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ar='16000').run(overwrite_output=True)

def transcribe_whisper(audio_path):
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result.get("segments", [])]

def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

class UnicodePDF(FPDF):
    def __init__(self, video_id, object_id, video_filename):
        super().__init__()
        self.video_id = video_id
        self.object_id = object_id
        self.video_filename = video_filename

    def header(self):
        self.set_font("DejaVu", size=12)
        self.cell(0, 10, "Subtitles Extracted from Video", ln=True, align="C")
        self.ln(5)
        self.set_font("DejaVu", size=10)
        self.cell(0, 10, f"Video ID: {self.video_id}", ln=True, align="C")
        self.cell(0, 10, f"Object ID: {self.object_id}", ln=True, align="C")
        self.cell(0, 10, f"Video Name: {self.video_filename}", ln=True, align="C")
        self.ln(5)



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

    # Extract only original filename without video_id prefix
    original_filename = "_".join(video_filename.split("_")[1:])

    audio_path = os.path.splitext(video_path)[0] + ".wav"
    extract_audio(video_path, audio_path)
    segments = transcribe_whisper(audio_path)
    os.remove(audio_path)

    pdf = UnicodePDF(video_id, object_id, original_filename)
    pdf.add_font("DejaVu", "", FONT_FILE, uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()

    for seg in segments:
        line = f"[{format_time(seg['start'])} - {format_time(seg['end'])}] {seg['text'].strip()}"
        pdf.multi_cell(0, 10, line)
        pdf.ln(2)

    pdf_filename = f"{os.path.splitext(original_filename)[0]}_subtitles_{video_id}.pdf"
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    pdf.output(pdf_path)

    insert_result = subtitles_collection.insert_one({
        "video_id": video_id,
        "video_object_id": ObjectId(object_id),
        "video_filename": original_filename,
        "pdf_filename": pdf_filename,
        "transcript": segments,
        "created_at": datetime.utcnow()
    })

    subtitle_object_id = insert_result.inserted_id
    print(f"Subtitle PDF ObjectId: {subtitle_object_id}")  # print in terminal

    return pdf_path, pdf_filename, subtitle_object_id






def extract_text_from_pdf_path(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found on server")
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def chunk_text(text, chunk_size=10000, chunk_overlap=1000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def store_embeddings_for_pdf(object_id: str, text_chunks):
    for chunk in text_chunks:
        uid = str(uuid.uuid4())
        embedding = embeddings_model.embed_documents([chunk])
        collection.add(
            documents=[chunk],
            embeddings=embedding,
            metadatas=[{"pdf_object_id": object_id}],
            ids=[uid]
        )





# ───────────── FastAPI App & Router Setup ─────────────
app = FastAPI(title="Video Subtitle & RAG PDF Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

router = APIRouter()





# ───────────── Video Upload and Processing Endpoints ─────────────

@router.post("/upload_video/")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file uploaded")

    video_id = str(uuid.uuid4())[:8]
    video_filename = f"{video_id}_{video.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)

    with open(video_path, "wb") as f:
        f.write(await video.read())

    result = video_collection.insert_one({
        "video_id": video_id,
        "video_filename": video_filename,
        "video_path": video_path,
        "uploaded_at": datetime.utcnow()
    })

    video_object_id = result.inserted_id  # Store ObjectId
    print(f"Video ObjectId: {video_object_id}")  # Print ObjectId to console/log

    return {
        "message": "Video uploaded successfully",
        "video_id": video_id,
        "object_id": str(video_object_id),
        "next_step": f"/process_from_file/{str(video_object_id)}"
    }


@router.post("/process_from_file/{object_id}")
def process_from_file(object_id: str):
    try:
        pdf_path, pdf_filename, subtitle_object_id = process_video_from_file(object_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "pdf_path": pdf_path,
        "pdf_filename": pdf_filename,
        "subtitle_object_id": str(subtitle_object_id)
    })





@router.get("/pdf/{object_id}")
def download_pdf(object_id: str):
    try:
        subtitle_doc = subtitles_collection.find_one({"_id": ObjectId(object_id)})
        if not subtitle_doc:
            raise HTTPException(status_code=404, detail="Subtitle document not found")
        pdf_path = os.path.join(PDF_FOLDER, subtitle_doc["pdf_filename"])
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not on server")
        return FileResponse(pdf_path, media_type="application/pdf", filename=subtitle_doc["pdf_filename"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transcript/{object_id}")
def get_transcript(object_id: str):
    try:
        doc = subtitles_collection.find_one({"_id": ObjectId(object_id)}, {"_id": 0, "transcript": 1})
        if not doc:
            raise HTTPException(status_code=404, detail="Transcript not found")
        return JSONResponse(content=doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/delete/{object_id}")
def delete_video_data(object_id: str):
    try:
        video_doc = video_collection.find_one({"_id": ObjectId(object_id)})
        if video_doc:
            video_path = video_doc.get("video_path")
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            video_collection.delete_one({"_id": ObjectId(object_id)})

        subtitle_doc = subtitles_collection.find_one({"video_id": video_doc['video_id']})
        if subtitle_doc:
            pdf_filename = subtitle_doc.get("pdf_filename")
            pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            subtitles_collection.delete_one({"_id": subtitle_doc["_id"]})

        return {"message": "Deleted successfully", "object_id": object_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/print_collections")
def print_collections():
    videos = list(video_collection.find())
    subtitles = list(subtitles_collection.find())

    def convert_object_ids(docs):
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            if "video_object_id" in doc:
                doc["video_object_id"] = str(doc["video_object_id"])
        return docs

    videos = convert_object_ids(videos)
    subtitles = convert_object_ids(subtitles)

    return {"videos": videos, "subtitles": subtitles}





# ───────────── PDF Embedding & RAG Question Answering Endpoints ─────────────

class QuestionRequest(BaseModel):
    question: str



@router.post("/api/load_pdf_mongo_to_chroma/{object_id}")
def load_pdf_mongo_to_chroma(object_id: str = Path(..., description="MongoDB ObjectId of the PDF subtitles document")):
    try:
        subtitle_doc = subtitles_collection.find_one({"_id": ObjectId(object_id)})
        if not subtitle_doc:
            raise HTTPException(status_code=404, detail="PDF document not found in DB")
        pdf_filename = subtitle_doc.get("pdf_filename")
        if not pdf_filename:
            raise HTTPException(status_code=404, detail="PDF filename not found in document")

        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        text = extract_text_from_pdf_path(pdf_path)
        chunks = chunk_text(text)
        store_embeddings_for_pdf(object_id, chunks)

        return {"message": f"PDF text embedded and stored in ChromaDB for ObjectId {object_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF from MongoDB: {str(e)}")







@router.post("/api/ask_question_from_video/{object_id}")
async def ask_question(object_id: str, request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    try:
        query_embedding = embeddings_model.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            where={"pdf_object_id": object_id}
        )
        docs = results.get("documents", [[]])[0]
        context = "\n\n".join(docs).strip()

        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Just give specific answer"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return {"answer": response.text.strip()}
    except Exception as e:
        raise HTTPException(500, f"Error generating answer: {e}")
    



#-------------------------------Global Knowledge--------------------------------
@router.post("/upload_global_pdf/")
async def upload_global_pdf(files: list[UploadFile] = File(...)):
    uploaded = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        pdf_id = str(uuid.uuid4())[:8]
        pdf_filename = f"{pdf_id}_{file.filename}"
        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)

        # Save PDF to disk
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Store in MongoDB
        result = global_pdfs_collection.insert_one({
            "pdf_id": pdf_id,
            "pdf_filename": pdf_filename,
            "pdf_path": pdf_path,
            "uploaded_at": datetime.utcnow()
        })
        object_id = result.inserted_id

        # Extract and Embed
        try:
            text = extract_text_from_pdf_path(pdf_path)
            chunks = chunk_text(text)

            for chunk in chunks:
                uid = str(uuid.uuid4())
                embedding = embeddings_model.embed_documents([chunk])
                global_knowledge_collection.add(
                    documents=[chunk],
                    embeddings=embedding,
                    metadatas=[{"pdf_object_id": str(object_id)}],
                    ids=[uid]
                )

            uploaded.append({
                "pdf_filename": pdf_filename,
                "object_id": str(object_id)
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {e}")

    return {
        "message": "Global PDFs processed and embedded",
        "uploaded": uploaded
    }



#------------global bot----------------

class GlobalQuestionRequest(BaseModel):
    question: str

@router.post("/ask_global_question/")
async def ask_global_question(request: GlobalQuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    try:
        query_embedding = embeddings_model.embed_query(question)
        results = global_knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )
        docs = results.get("documents", [[]])[0]
        context = "\n\n".join(docs).strip()

        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Just give specific answer."""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return {"answer": response.text.strip()}

    except Exception as e:
        raise HTTPException(500, f"Error answering from global knowledge: {e}")





# ───────────── Root Route: Redirect to chat.html ─────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/chat.html")

app.include_router(router)

