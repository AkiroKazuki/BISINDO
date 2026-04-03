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

class ConsecutiveTriggerLimiter:
    """Requires N consecutive hits within a time window to trigger.
    
    This acts as a high-confidence threshold for critical alerts like Voice Calls.
    """
    def __init__(self, required_hits: int = 2, window_seconds: float = 60.0):
        self.required_hits = required_hits
        self.window_seconds = window_seconds
        self.history: List[float] = []

    def should_trigger(self) -> bool:
        now = time.time()
        self.history = [t for t in self.history if now - t < self.window_seconds]
        self.history.append(now)
        # Only trigger when we strictly reach the required hits
        return len(self.history) >= self.required_hits

voice_rate_limiter = ConsecutiveTriggerLimiter(required_hits=2, window_seconds=60.0)

# --- Web Push subscription storage ---
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

# --- Twilio WhatsApp Message ---

def send_whatsapp_twilio(contact_number: str, gesture: str, user_name: str) -> Dict:
    """Send a WhatsApp message via Twilio.
    Bypasses telecom SMS/VOIP blocking by transmitting over data.
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("WhatsApp not sent -- Twilio credentials not configured")
        return {"status": "disabled", "detail": "Twilio credentials not set"}

    # Format local numbers to E.164
    if contact_number.startswith("0"):
        contact_number = "+62" + contact_number[1:]

    # Require multiple confirmation triggers within 60s for high-confidence alert
    if not voice_rate_limiter.should_trigger():
        logger.info("WhatsApp deferred -- waiting for 2nd confirmation to prevent false alarms.")
        return {"status": "rate_limited", "detail": "Waiting for 2nd confirmation within 60s"}

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Twilio WhatsApp requires 'whatsapp:' prefix
        body = f"🚨 *DARURAT BISINDO*\n\n{user_name} membutuhkan bantuan segera!\nIsyarat terdeteksi: *{gesture}*"
        
        message = client.messages.create(
            body=body,
            to=f"whatsapp:{contact_number}",
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}" if TWILIO_PHONE_NUMBER else "whatsapp:+14155238886"
        )

        logger.info(f"Twilio WhatsApp initiated to {contact_number} (SID: {message.sid})")
        return {"status": "sent", "detail": {"sid": message.sid}}
    except Exception as e:
        logger.error(f"Twilio WhatsApp failed: {e}")
        return {"status": "failed", "detail": str(e)}


# --- CallMeBot WhatsApp Voice Call ---

CALLMEBOT_API_KEY = os.getenv("CALLMEBOT_API_KEY")

def make_callmebot_voice_call(contact_number: str, gesture: str, user_name: str) -> Dict:
    """Make a WhatsApp Voice Call via free CallMeBot API."""
    if not CALLMEBOT_API_KEY:
        logger.warning("CallMeBot voice call disabled -- CALLMEBOT_API_KEY not set in .env")
        return {"status": "disabled", "detail": "No API key"}

    if contact_number.startswith("0"):
        contact_number = "+62" + contact_number[1:]

    # CallMeBot API uses GET request with urlencoded text
    import urllib.parse
    text = f"Peringatan Darurat. {user_name} membutuhkan bantuan. Isyarat {gesture} terdeteksi."
    encoded_text = urllib.parse.quote(text)
    
    url = f"https://api.callmebot.com/whatsapp.php?phone={contact_number}&text={encoded_text}&apikey={CALLMEBOT_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        logger.info(f"CallMeBot initiated to {contact_number}")
        return {"status": "initiated", "detail": response.text}
    except Exception as e:
        logger.error(f"CallMeBot failed: {e}")
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
        "twilio_wa_status": send_whatsapp_twilio(contact_number, gesture, user_name),
        "callmebot_status": make_callmebot_voice_call(contact_number, gesture, user_name),
        "push_status": send_push_notification(gesture, user_name),
    }

    logger.info(f"All notifications triggered for gesture: {gesture}")
    return result
