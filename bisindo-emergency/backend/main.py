"""
FastAPI backend for BISINDO Emergency Detection System.

Endpoints:
  WebSocket /ws          — Real-time keypoint streaming + inference
  POST     /notify       — Trigger notification pipeline
  POST     /subscribe-push — Store Web Push subscription
  GET      /health       — Health check

Usage:
    cd bisindo-emergency
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import InferencePipeline
from backend.notification import (
    trigger_all_notifications,
    store_push_subscription,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="BISINDO Emergency Detection API",
    description="Real-time sign language emergency detection using ST-GCN",
    version="1.0.0"
)

# CORS (allow frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference pipeline
pipeline = InferencePipeline()


# --- Pydantic models ---

class NotifyRequest(BaseModel):
    gesture: str
    contact_number: str
    user_name: str


class PushSubscription(BaseModel):
    endpoint: str
    keys: dict


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": pipeline.model_loaded
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time keypoint streaming and inference.

    Client sends: { "keypoints": [[x,y,z], ...] } — 75 landmarks
    Server sends: { "class": "TOLONG", "confidence": 0.92, "is_confirmed": false, ... }
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive keypoint data
            data = await ws.receive_text()
            message = json.loads(data)

            keypoints_raw = message.get("keypoints", [])

            # Validate
            if len(keypoints_raw) != 75:
                await ws.send_json({
                    "error": f"Expected 75 keypoints, got {len(keypoints_raw)}"
                })
                continue

            # Convert to numpy
            keypoints = np.array(keypoints_raw, dtype=np.float32)

            if keypoints.shape != (75, 3):
                await ws.send_json({
                    "error": f"Expected shape (75, 3), got {keypoints.shape}"
                })
                continue

            # Process through inference pipeline
            result = pipeline.process_frame(keypoints)

            # Send result back
            await ws.send_json({
                "class": result['class'],
                "confidence": round(result['confidence'], 4),
                "is_confirmed": result['is_confirmed'],
                "in_cooldown": result['in_cooldown'],
                "buffer_full": result['buffer_full'],
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.post("/notify")
async def notify(request: NotifyRequest):
    """Trigger all notification channels (SMS, Voice, Push).

    Body: { "gesture": "TOLONG", "contact_number": "+628xxx", "user_name": "..." }
    """
    result = trigger_all_notifications(
        gesture=request.gesture,
        contact_number=request.contact_number,
        user_name=request.user_name
    )

    return result


@app.post("/subscribe-push")
async def subscribe_push(subscription: PushSubscription):
    """Store a Web Push subscription from the browser.

    Body: { "endpoint": "https://...", "keys": { "p256dh": "...", "auth": "..." } }
    """
    store_push_subscription(subscription.model_dump())
    return {"status": "subscribed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
