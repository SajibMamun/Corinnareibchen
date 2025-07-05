import os
import time
import ffmpeg
import whisper

whisper_model = whisper.load_model("base") #medium can be use

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
    for file in os.listdir("uploads"):
        path = os.path.join("uploads", file)
        if os.path.isfile(path):
            os.remove(path)