from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
import os
import shutil

app = FastAPI()

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

templates = Jinja2Templates(directory="templates")

# Global variables for progress and logs
progress = 0
logs = []
video_path = ""


@app.get("/")
def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/process")
async def process_video(video: UploadFile, model: str = Form(...)):
    global progress, logs, video_path
    logs = []  # Clear previous logs
    progress = 0  # Reset progress

    # Save uploaded video
    video_path = f"uploads/{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Fake processing simulation (replace this with actual code execution)
    output_path = f"outputs/output_{video.filename}"

    # Simulate long-running process with logs
    for i in range(1, 101):
        time.sleep(0.05)  # Simulate processing delay
        progress = i
        logs.append(f"Processing frame {i}/100\n")

    # Replace this with the actual command or script execution
    os.system(f"python run_model.py {video_path} {output_path} {model}")

    return {"success": True, "output_path": output_path}


@app.get("/progress")
async def get_progress():
    global progress
    return JSONResponse(content={"progress": progress})


@app.get("/logs")
async def get_logs():
    global logs
    return JSONResponse(content={"logs": logs})


import cv2
from fastapi.responses import StreamingResponse

# Streaming generator for frames
def generate_frames():
    global video_path
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Encode frame as JPEG for faster streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.get("/stream")
async def stream_video():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
    
    
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/log-stream")
async def log_stream():
    async def event_stream():
        global logs
        last_index = 0
        while True:
            # Send new logs only
            if last_index < len(logs):
                for i in range(last_index, len(logs)):
                    yield f"data: {logs[i]}\n\n"
                last_index = len(logs)
            await asyncio.sleep(0.5)  # Send updates every 500ms

    return StreamingResponse(event_stream(), media_type="text/event-stream")

    
 
    
    



