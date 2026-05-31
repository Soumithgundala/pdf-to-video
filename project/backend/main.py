"""
Manga Recap Video Pipeline - Main Entry Point
Run the FastAPI backend server.
"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("UVICORN_RELOAD", "").lower() in {"1", "true", "yes"},
        log_level="info"
    )
