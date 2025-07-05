from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from bson import ObjectId
from datetime import datetime
import os
import threading
import uuid
from services.processor import process_video_from_file
from utils.helpers import clean_uploads
from db.mongo import video_collection, subtitles_collection


router = APIRouter()


UPLOAD_FOLDER = "uploads"
PDF_FOLDER = "pdfs"



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

    return {
        "message": "Video uploaded successfully",
        "video_id": video_id,
        "object_id": str(result.inserted_id),
        "next_step": f"/process_from_file/{str(result.inserted_id)}"
    }




@router.post("/process_from_file/{object_id}")
def process_from_file(object_id: str):
    try:
        pdf_path, pdf_filename = process_video_from_file(object_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    threading.Thread(target=clean_uploads, daemon=True).start()
    return FileResponse(pdf_path, filename=pdf_filename, media_type="application/pdf")



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
