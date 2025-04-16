import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
DATA_DIR.mkdir(exist_ok=True)

FAQ_FILE = DATA_DIR / "faqs.csv"  # Use the converted CSV file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

APP_TITLE = "Nawatech FAQ Chatbot"
APP_ICON = "🤖"
APP_DESCRIPTION = "Ada yang ingin ditanyakan tentang Nawatech? Silakan bertanya!"  # Indonesian prompt

# Model settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")  # OpenAI API URL
EMBEDDING_MODEL = "ebbge-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen2.5-7B")  
TEMPERATURE = 0.1
TOP_K = 6 
LLM_API_KEY = os.getenv("LLM_API_KEY")

# UI Settings
MAX_HISTORY_LENGTH = 20  # Maximum number of messages to keep in history