import os
import sys

# Add backend directory to path so we can import notification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from notification import make_voice_call, voice_rate_limiter

# Force limiter to trigger immediately for test
voice_rate_limiter.required_hits = 1

# Provide your actual verified number here (or pass via arg)
test_number = sys.argv[1] if len(sys.argv) > 1 else "+628123456789"

print(f"Testing Twilio call to: {test_number}")
response = make_voice_call(test_number, "KEBAKARAN", "Test User")
print("Twilio Response:", response)
