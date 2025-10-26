"""
config.py

Centralized configuration loaded from a .env file (if present).
Expose keys and simple defaults used by main.py.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def getenv(key: str, default=None):
    v = os.getenv(key)
    return v if v is not None else default

# OpenAI / LLM
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
DEFAULT_MODEL = getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SYSTEM_PROMPT = getenv("DEFAULT_SYSTEM_PROMPT", "")
DEFAULT_SCHEMA_DOC = getenv("DEFAULT_SCHEMA_DOC", "")

# IMGBB (image upload)
IMGBB_API_KEY = getenv("IMGBB_API_KEY")

# Optional Telegram / notification
TELEGRAM_CHAT_ID = getenv("TELEGRAM_CHAT_ID")

# App defaults
OUTPUT_DIR = getenv("OUTPUT_DIR", str(ROOT / "outputs"))

def as_dict():
    return {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "DEFAULT_MODEL": DEFAULT_MODEL,
        "IMGBB_API_KEY": IMGBB_API_KEY,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "OUTPUT_DIR": OUTPUT_DIR,
    }
