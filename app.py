import os
import uuid
from datetime import datetime
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Path, Body
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
from uuid import uuid4
from datetime import datetime
from typing import List
import random
import string

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

    # Global chat sessions and messages collections
    global_chat_collection = db["global_Chat_history"]       # stores sessions
    global_chat_messages_collection = db["global_Chat_messages"]  # stores messages per session

except errors.ServerSelectionTimeoutError as err:
    raise Exception(f"Could not connect to MongoDB: {err}")

# ───────────── Whisper Model Setup ─────────────
whisper_model = whisper.load_model("base")

# ───────────── ChromaDB Setup ─────────────
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "video_pdf_knowledge"

GLOBAL_COLLECTION_NAME = "global_pdf_knowledge"
VIDEO_CHAT_COLLECTION = "video_chat_memory"

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
global_knowledge_collection = chroma_client.get_or_create_collection(name=GLOBAL_COLLECTION_NAME)
video_chat_collection = chroma_client.get_or_create_collection(name=VIDEO_CHAT_COLLECTION)


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
    session_id: str  # <-- Add this line


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
    






def store_chat_message_in_chroma(role: str, content: str, session_id: str, video_object_id: str):
    embedding = embeddings_model.embed_documents([content])
    uid = str(uuid4())
    video_chat_collection.add(
        documents=[content],
        embeddings=embedding,
        metadatas=[{
            "role": role,
            "session_id": session_id,
            "video_object_id": video_object_id
        }],
        ids=[uid]
    )





def get_chat_history_from_chroma(session_id: str, video_object_id: str, n_messages: int = 6):
    results = video_chat_collection.get(
        where={"session_id": session_id}
    )

    if not results or "documents" not in results:
        return ""

    # Reconstruct chat turns and sort by insertion order (Chroma preserves it)
    docs = results["documents"]
    metas = results["metadatas"]

    messages = []
    for doc, meta in zip(docs, metas):
        if meta.get("video_object_id") == video_object_id:
            role = meta.get("role", "user").capitalize()
            messages.append(f"{role}: {doc.strip()}")

    # Return last N messages
    print(messages[-n_messages:])
    return "\n".join(messages[-n_messages:])








def generate_session_id(length=12):
    """Generates a random alphanumeric session ID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@router.post("/api/create_video_chat_session")
def create_video_session():
    session_id = generate_session_id()
    return {"session_id": session_id, "message": "Session created successfully"}



@router.post("/api/ask_question_from_video/{object_id}")
async def ask_question(object_id: str, request: QuestionRequest):
    question = request.question.strip()
    session_id = request.session_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        # 1️⃣ Embed the question to retrieve relevant subtitles
        query_embedding = embeddings_model.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            where={"pdf_object_id": object_id}
        )
        docs = results.get("documents", [[]])[0]
        context = "\n\n".join(docs).strip()

        # 2️⃣ Retrieve chat memory (limit to last 10 messages)
        history = get_chat_history_from_chroma(session_id, object_id, n_messages=10)


        prompt = f"""You are a helpful assistant.You are a helpful assistant answering user questions based on previous chat history and video subtitles.

Video Knowledge:
{context}

Your previous Chat History of this user:
{history}

User: {question}
Assistant: Provide a direct, clear, and specific answer only. Don't print whole knowledgebase  and chat history if user asked"""

        # 5️⃣ Generate answer using Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # 6️⃣ Store the interaction in ChromaDB memory
        store_chat_message_in_chroma("user", question, session_id, object_id)
        store_chat_message_in_chroma("assistant", answer, session_id, object_id)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    

@router.delete("/api/clear_session_chat/{session_id}")
def clear_session_chat(session_id: str):
    try:
        video_chat_collection.delete(where={"session_id": session_id})
        return {"message": f"Chat history for session {session_id} cleared."}
    except Exception as e:
        raise HTTPException(500, f"Error clearing session: {str(e)}")






# ---------------- Global Knowledge Chat Sessions & Messages -----------------

class GlobalQuestionRequest(BaseModel):
    question: str
    session_id: str

def serialize_session(session_doc):
    return {
        "_id": str(session_doc["_id"]),
        "name": session_doc.get("name", "Unnamed Session"),
    }

def serialize_message(msg_doc):
    return {
        "role": msg_doc.get("role"),
        "content": msg_doc.get("content"),
    }



@router.get("/api/list_sessions")
def list_sessions():
    sessions = list(global_chat_collection.find({}, {"name": 1, "created_at": 1}))
    sessions = sorted(sessions, key=lambda x: x.get("created_at", datetime.min), reverse=True)
    return [serialize_session(s) for s in sessions]

@router.post("/api/create_session")
def create_session():
    new_session = {
        "name": "New Session",
        "created_at": datetime.utcnow()
    }
    result = global_chat_collection.insert_one(new_session)
    new_session["_id"] = result.inserted_id
    return serialize_session(new_session)





@router.patch("/api/rename_session/{session_id}")
def rename_session(session_id: str, data: dict = Body(...)):
    new_name = data.get("name")
    if not new_name:
        raise HTTPException(400, "Name cannot be empty")

    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")

    update_result = global_chat_collection.update_one(
        {"_id": oid},
        {"$set": {"name": new_name}}
    )
    if update_result.matched_count == 0:
        raise HTTPException(404, "Session not found")

    return {"message": "Renamed successfully"}





@router.delete("/api/delete_session/{session_id}")
def delete_session(session_id: str):
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")

    delete_result = global_chat_collection.delete_one({"_id": oid})
    if delete_result.deleted_count == 0:
        raise HTTPException(404, "Session not found")

    # Also delete all messages in that session
    global_chat_messages_collection.delete_many({"session_id": oid})

    return {"message": "Deleted successfully"}













@router.get("/api/session_messages/{session_id}")
def get_session_messages(session_id: str):
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")

    messages = list(global_chat_messages_collection.find({"session_id": oid}).sort("timestamp", 1))
    return [serialize_message(m) for m in messages]

@router.post("/ask_global_question/")
async def ask_global_question(request: GlobalQuestionRequest):
    question = request.question.strip()
    session_id = request.session_id.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    if not session_id:
        raise HTTPException(400, "Session ID required.")

    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID")

    try:
        # Embed query and search knowledge base
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

        answer = response.text.strip()

        # Save user question message
        global_chat_messages_collection.insert_one({
            "session_id": oid,
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow()
        })
        # Save bot answer message
        global_chat_messages_collection.insert_one({
            "session_id": oid,
            "role": "bot",
            "content": answer,
            "timestamp": datetime.utcnow()
        })

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(500, f"Error answering from global knowledge: {e}")














# ───────────── Root Route: Redirect to chat.html ─────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/chat.html")

app.include_router(router)
