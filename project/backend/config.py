"""
Configuration settings for the Manga Recap Video Pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

# Load backend-local env first, then allow project-root env to override it.
load_dotenv(BASE_DIR / ".env")
load_dotenv(ROOT_DIR / ".env", override=True)

def _resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


WORKSPACE_DIR = _resolve_local_path(os.getenv("WORKSPACE_DIR", "temp_workspace"))

# Local SQLite database
DB_PATH = str(_resolve_local_path(os.getenv("DB_PATH", "manga_pipeline.db")))

# PDF resolution (DPI) for extraction (lowering to 150 speeds up pipeline significantly)
PDF_DPI = int(os.getenv("PDF_DPI", "150"))

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", os.getenv("VITE_SUPABASE_URL", ""))
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", os.getenv("VITE_SUPABASE_ANON_KEY", ""))
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    SUPABASE_ANON_KEY,
)
SUPABASE_ENABLED = bool(
    SUPABASE_URL
    and SUPABASE_SERVICE_ROLE_KEY
    and SUPABASE_SERVICE_ROLE_KEY != SUPABASE_ANON_KEY
)

# LLM APIs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Video settings
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 24
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg")
VIDEO_CODEC = os.getenv("VIDEO_CODEC", "h264_nvenc")
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "fast")
VIDEO_FALLBACK_CODEC = os.getenv("VIDEO_FALLBACK_CODEC", "libx264")
VIDEO_FALLBACK_PRESET = os.getenv("VIDEO_FALLBACK_PRESET", "medium")
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "5000k")
VIDEO_ENCODE_STALL_TIMEOUT_SECONDS = int(os.getenv("VIDEO_ENCODE_STALL_TIMEOUT_SECONDS", "300"))
VIDEO_ENCODE_STALL_CHECK_SECONDS = int(os.getenv("VIDEO_ENCODE_STALL_CHECK_SECONDS", "15"))
TARGET_AUDIO_DURATION_SECONDS = 75
TARGET_SCRIPT_WORD_COUNT = (110, 140)
PANELS_PER_PART = (5, 7)

# Audio
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+8%")
BACKGROUND_MUSIC_VOLUME = float(os.getenv("BACKGROUND_MUSIC_VOLUME", "0.1"))
