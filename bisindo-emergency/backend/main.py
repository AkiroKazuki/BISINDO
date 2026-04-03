"""
FastAPI backend for BISINDO Emergency Detection System.

Endpoints:
  WebSocket /ws              — Real-time keypoint streaming + inference
  POST     /notify           — Trigger notification pipeline
  POST     /subscribe-push   — Store Web Push subscription
  GET      /health           — Health check
  GET      /vapid-public-key — Return VAPID public key for push subscription

Usage:
    cd bisindo-emergency
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

VAPID_PUBLIC_KEY = os.getenv('VAPID_PUBLIC_KEY', '')

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
    title="Sistem Deteksi Isyarat Darurat BISINDO (ST-GCN)",
    description="PENGEMBANGAN SISTEM DETEKSI ISYARAT DARURAT BISINDO BERBASIS ARSITEKTUR ST-GCN DENGAN PENGUJIAN KETAHANAN TERHADAP OKLUSI PARSIAL",
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
    location_url: Optional[str] = None


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

            # [DIAGNOSTIC] Log hand keypoint detection (once per connection)
            if not hasattr(websocket_endpoint, '_diag_logged'):
                hand_kps = keypoints[33:75]  # left + right hand
                nonzero_hands = int((np.abs(hand_kps).sum(axis=1) > 0.001).sum())
                logger.info(f"[DIAGNOSTIC] Hand keypoints detected: {nonzero_hands}/42"
                            f" (pose: 33/33, total non-zero: "
                            f"{int((np.abs(keypoints).sum(axis=1) > 0.001).sum())}/75)")
                if nonzero_hands == 0:
                    logger.warning("[DIAGNOSTIC] NO HAND LANDMARKS DETECTED! "
                                   "The model needs hand data to distinguish gestures.")
                websocket_endpoint._diag_logged = True

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
        user_name=request.user_name,
        location_url=request.location_url
    )

    return result


@app.post("/subscribe-push")
async def subscribe_push(subscription: PushSubscription):
    """Store a Web Push subscription from the browser.

    Body: { "endpoint": "https://...", "keys": { "p256dh": "...", "auth": "..." } }
    """
    store_push_subscription(subscription.model_dump())
    return {"status": "subscribed"}


@app.get("/vapid-public-key")
async def vapid_public_key():
    """Return the VAPID public key for push notification subscription."""
    if not VAPID_PUBLIC_KEY:
        logger.warning("VAPID_PUBLIC_KEY is not configured in .env")
        return {"publicKey": "", "error": "VAPID key not configured"}
    return {"publicKey": VAPID_PUBLIC_KEY}

# --- Mount Static Frontend ---
# This serves the React build folder so that LocalTunnel can expose 
# both the Frontend and Backend via a single port (8000).
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    logger.info(f"Mounted static frontend from {frontend_dist}")
else:
    logger.warning("Frontend dist not found. Run 'npm run build' in the frontend directory.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
