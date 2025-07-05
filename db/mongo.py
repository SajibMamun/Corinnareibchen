from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["Corinnareibchen"]
video_collection = db["video"]
subtitles_collection = db["subtitles"]
