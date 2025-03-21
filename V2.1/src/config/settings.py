"""
System-wide settings and configurations.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-70b-chat-hf")
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# Analysis Configuration
DEFAULT_FRAMEWORK = "ICAP"  # Can be changed to other frameworks
SENTIMENT_THRESHOLD = 0.5
QUALITY_THRESHOLD = 0.7

# Evaluation Configuration
EVALUATION_METRICS = [
    "bertscore",
    "lime",
    "shap",
    "llm_evaluation"
]

# Data Processing Configuration
CSV_ENCODING = "utf-8"
DATE_FORMAT = "%Y-%m-%d"

# Visualization Configuration
PLOT_THEME = "plotly_white"
COLOR_SCHEME = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "highlight": "#3498db"
}

# API Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True 