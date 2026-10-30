import base64
import os
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import MODEL_PATH, LIVE_MODEL_PATH
from backend.services.thermal_service import thermal_image_service
from backend.services.video_service import thermal_video_service
from backend.services.live_service import live_processor

app = FastAPI(
    title="Thermal FLIR Border Surveillance API",
    description="FastAPI Backend for Thermal Person Detection, Behavior Tracking, and Live Stream Surveillance",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "thermal_model_exists": os.path.exists(MODEL_PATH),
        "live_model_exists": os.path.exists(LIVE_MODEL_PATH),
        "service": "Thermal Border Surveillance API"
    }

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    apply_clahe: bool = Query(False)
):
    try:
        content = await file.read()
        result = thermal_image_service.process_image(
            content, conf=conf, apply_clahe=apply_clahe
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Query(0.15, ge=0.01, le=1.0)
):
    try:
        content = await file.read()
        task = thermal_video_service.create_task(
            filename=file.filename or "video.mp4",
            content=content,
            conf_threshold=conf
        )
        return {
            "task_id": task.task_id,
            "status": task.status,
            "message": "Video tracking and behavior classification task queued."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detect/video/status/{task_id}")
async def get_video_status(task_id: str):
    task = thermal_video_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": round(task.progress * 100, 1),
        "alert_count": task.alert_count,
        "tracks_count": task.tracks_count,
        "total_frames": task.total_frames,
        "processed_frames": task.processed_frames,
        "duration_sec": task.duration_sec,
        "fps": task.fps,
        "classification_breakdown": task.classification_breakdown,
        "confirmed_alerts": task.confirmed_alerts,
        "tracked_entities": task.tracked_entities,
        "non_human_summary": task.non_human_summary,
        "error_message": task.error_message,
        "result_url": task.result_url
    }

@app.api_route("/api/detect/video/result/{task_id}", methods=["GET", "HEAD"])
async def get_video_result(task_id: str):
    task = thermal_video_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task has not finished processing yet")
    if not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="Processed video file not found")

    return FileResponse(
        task.output_path,
        media_type="video/mp4",
        filename="surveillance_annotated.mp4"
    )

@app.post("/api/detect/live-frame")
async def detect_live_frame(
    file: UploadFile = File(...),
    conf: float = Query(0.55, ge=0.01, le=1.0)
):
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Invalid frame image")
        res = live_processor.process_frame(frame_bgr, conf_threshold=conf)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/api/ws/live")
async def live_stream_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Expected format: {"image": "data:image/jpeg;base64,...", "conf": 0.55}
            image_b64 = data.get("image", "")
            conf = float(data.get("conf", 0.55))

            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]

            if not image_b64:
                continue

            raw_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is not None:
                result = live_processor.process_frame(frame_bgr, conf_threshold=conf)
                await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

# Serve Angular frontend if built
dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "frontend" / "browser"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")
