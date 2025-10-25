"""
API Keys and Configuration Management
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_MOCK_AI_ENV = os.getenv("USE_MOCK_AI", "True").lower()
USE_MOCK_AI = USE_MOCK_AI_ENV in ["true", "1", "yes", "y"]

if not USE_MOCK_AI and not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not found but USE_MOCK_AI=False")
    print("🔄 Falling back to mock mode")
    USE_MOCK_AI = True

if USE_MOCK_AI:
    print("🧠 Running in Mock AI Mode (no real API calls).")
    print("💡 To use real OpenAI: Set USE_MOCK_AI=False in .env")
else:
    if OPENAI_API_KEY:
        masked_key = f"{OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:]}"
        print("✅ OpenAI API Key loaded successfully!")
        print(f"🔑 Key: {masked_key}")
        print("✅ Mock AI disabled — Using real OpenAI responses")
    else:
        print("❌ OpenAI API Key missing!")
        USE_MOCK_AI = True

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS_PER_REQUEST = int(os.getenv("MAX_TOKENS_PER_REQUEST", "400"))

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ["true", "1", "yes"]
FALLBACK_TO_MOCK = os.getenv("FALLBACK_TO_MOCK", "True").lower() in ["true", "1", "yes"]
LOG_API_ERRORS = os.getenv("LOG_API_ERRORS", "True").lower() in ["true", "1", "yes"]

ENABLE_REAL_TIME_ADVICE = os.getenv("ENABLE_REAL_TIME_ADVICE", "True").lower() in [
    "true",
    "1",
    "yes",
]
ENABLE_MARKET_DATA = os.getenv("ENABLE_MARKET_DATA", "False").lower() in [
    "true",
    "1",
    "yes",
]
ENABLE_BANKING_INTEGRATION = os.getenv(
    "ENABLE_BANKING_INTEGRATION", "False"
).lower() in ["true", "1", "yes"]

DATABASE_URL = os.getenv("DATABASE_URL")


def validate_configuration():
    issues = []
    if not USE_MOCK_AI and not OPENAI_API_KEY:
        issues.append("OpenAI API key missing (required when USE_MOCK_AI=False)")
    if issues:
        print("⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    print("✅ Configuration ready for deployment!")
    return True


validate_configuration()
