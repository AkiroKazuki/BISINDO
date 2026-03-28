"""
Notification pipeline for BISINDO Emergency System.

Supports:
- SMS via Textbee.dev (free, uses Android SIM card as gateway)
- Voice call via Twilio (trial, $15 free credit)
- Web Push via pywebpush (VAPID-authenticated)

All credentials loaded from .env. If not configured, each service logs a clear
WARNING -- no silent failures.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables (explicit path for CWD-independence)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(_env_path)

# Textbee config
TEXTBEE_API_KEY = os.getenv("TEXTBEE_API_KEY")
TEXTBEE_DEVICE_ID = os.getenv("TEXTBEE_DEVICE_ID")

# Twilio config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# VAPID config (Web Push)
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
VAPID_MAILTO = os.getenv("VAPID_MAILTO", "mailto:admin@example.com")


def _check_and_warn():
    """Log warnings for unconfigured notification services at startup."""
    if not TEXTBEE_API_KEY or not TEXTBEE_DEVICE_ID:
        logger.warning("TEXTBEE_API_KEY or TEXTBEE_DEVICE_ID not set -- SMS notifications disabled")

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        logger.warning("TWILIO_ACCOUNT_SID/AUTH_TOKEN/PHONE_NUMBER not set -- Voice call notifications disabled")

    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        logger.warning("VAPID_PRIVATE_KEY or VAPID_PUBLIC_KEY not set -- Web Push notifications disabled")


# Run warnings on import
_check_and_warn()


# --- Rate limiting ---

class RateLimiter:
    """Simple rate limiter for voice calls.

    Prevents spam: only trigger voice call if the same gesture
    is confirmed > 3 times in 30 seconds.
    """

    def __init__(self, max_triggers: int = 3, window_seconds: float = 30.0):
        self.max_triggers = max_triggers
        self.window_seconds = window_seconds
        self.history: List[float] = []

    def should_trigger(self) -> bool:
        """Check if a voice call should be triggered."""
        now = time.time()
        # Remove old entries
        self.history = [t for t in self.history if now - t < self.window_seconds]
        self.history.append(now)

        return len(self.history) > self.max_triggers


voice_rate_limiter = RateLimiter(max_triggers=3, window_seconds=30.0)

# --- Push subscription storage ---
push_subscriptions: List[dict] = []


def store_push_subscription(subscription: dict):
    """Store a Web Push subscription from the browser."""
    # Avoid duplicates
    for existing in push_subscriptions:
        if existing.get("endpoint") == subscription.get("endpoint"):
            return
    push_subscriptions.append(subscription)
    logger.info(f"Push subscription stored (total: {len(push_subscriptions)})")


# --- SMS via Textbee ---

def send_sms_textbee(contact_number: str, gesture: str, user_name: str) -> Dict:
    """Send SMS via Textbee.dev API.

    Args:
        contact_number: Recipient phone number (e.g., "+628xxx").
        gesture: Detected gesture name.
        user_name: Name of the user in distress.

    Returns:
        Dict with 'status' and 'detail' keys.
    """
    if not TEXTBEE_API_KEY or not TEXTBEE_DEVICE_ID:
        logger.warning("SMS not sent -- TEXTBEE credentials not configured")
        return {"status": "disabled", "detail": "Textbee credentials not set"}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        f"DARURAT: {user_name} membutuhkan bantuan segera.\n"
        f"Isyarat [{gesture}] terdeteksi pada [{timestamp}].\n"
        f"Hubungi segera."
    )

    url = f"https://api.textbee.dev/api/v1/gateway/devices/{TEXTBEE_DEVICE_ID}/sendSMS"
    headers = {
        "x-api-key": TEXTBEE_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "receivers": [contact_number],
        "message": message
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"SMS sent to {contact_number} via Textbee")
        return {"status": "sent", "detail": response.json()}
    except requests.RequestException as e:
        logger.error(f"SMS failed: {e}")
        return {"status": "failed", "detail": str(e)}


# --- Voice Call via Twilio ---

def make_voice_call(contact_number: str, gesture: str, user_name: str) -> Dict:
    """Make a voice call via Twilio.

    Rate-limited: only triggers if same gesture confirmed >3 times in 30s.

    Args:
        contact_number: Recipient phone number.
        gesture: Detected gesture name.
        user_name: Name of the user in distress.

    Returns:
        Dict with 'status' and 'detail' keys.
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        logger.warning("Voice call not made -- Twilio credentials not configured")
        return {"status": "disabled", "detail": "Twilio credentials not set"}

    # Rate limiting
    if not voice_rate_limiter.should_trigger():
        return {"status": "rate_limited", "detail": "Not enough repeated confirmations"}

    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        twiml = (
            f'<Response>'
            f'<Say language="id-ID">Peringatan darurat. {user_name} membutuhkan bantuan segera. '
            f'Isyarat {gesture} terdeteksi.</Say>'
            f'</Response>'
        )

        call = client.calls.create(
            twiml=twiml,
            to=contact_number,
            from_=TWILIO_PHONE_NUMBER
        )

        logger.info(f"Voice call initiated to {contact_number} (SID: {call.sid})")
        return {"status": "initiated", "detail": {"sid": call.sid}}
    except Exception as e:
        logger.error(f"Voice call failed: {e}")
        return {"status": "failed", "detail": str(e)}


# --- Web Push Notification ---

def send_push_notification(gesture: str, user_name: str) -> Dict:
    """Send Web Push notification to all subscribed browsers.

    Args:
        gesture: Detected gesture name.
        user_name: Name of the user in distress.

    Returns:
        Dict with 'status' and 'detail' keys.
    """
    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        logger.warning("Push notification not sent -- VAPID keys not configured")
        return {"status": "disabled", "detail": "VAPID keys not set"}

    if not push_subscriptions:
        logger.warning("No push subscriptions registered")
        return {"status": "no_subscriptions", "detail": "No browsers subscribed"}

    try:
        from pywebpush import webpush, WebPushException

        data = json.dumps({
            "title": "DARURAT",
            "body": f"{user_name} membutuhkan bantuan -- isyarat {gesture} terdeteksi",
            "icon": "/icon.png"
        })

        vapid_claims = {"sub": VAPID_MAILTO}
        sent_count = 0

        for subscription in push_subscriptions:
            try:
                webpush(
                    subscription_info=subscription,
                    data=data,
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims=vapid_claims
                )
                sent_count += 1
            except WebPushException as e:
                logger.error(f"Push failed for one subscription: {e}")

        logger.info(f"Push notification sent to {sent_count}/{len(push_subscriptions)} subscribers")
        return {"status": "sent", "detail": {"sent": sent_count, "total": len(push_subscriptions)}}
    except Exception as e:
        logger.error(f"Push notification failed: {e}")
        return {"status": "failed", "detail": str(e)}


# --- Unified notification trigger ---

def trigger_all_notifications(gesture: str, contact_number: str,
                              user_name: str) -> Dict:
    """Trigger all notification channels.

    Args:
        gesture: Detected gesture name.
        contact_number: Emergency contact phone number.
        user_name: Name of the user in distress.

    Returns:
        Dict with status for each notification channel.
    """
    result = {
        "sms_status": send_sms_textbee(contact_number, gesture, user_name),
        "call_status": make_voice_call(contact_number, gesture, user_name),
        "push_status": send_push_notification(gesture, user_name),
    }

    logger.info(f"All notifications triggered for gesture: {gesture}")
    return result
