# Manga Recap Video Pipeline

Automated pipeline that transforms manga chapter PDFs into 4-part video miniseries with AI-generated voiceovers and cinematic effects.

## Tech Stack

- **PDF/Image Processing**: pdf2image, OpenCV, Pillow
- **AI/LLM**: OpenAI GPT-4o or Google Gemini 1.5 Pro
- **Audio**: edge-tts (free, high-quality TTS)
- **Video**: MoviePy with Ken Burns effects
- **Backend**: FastAPI
- **Database**: Supabase (PostgreSQL with RLS)
- **Frontend**: React + TypeScript + Tailwind

## Project Structure

```
project/
├── backend/
│   ├── modules/
│   │   ├── pdf_processor.py      # Phase 1: PDF to pages
│   │   ├── panel_extractor.py   # Phase 1: Panel extraction
│   │   ├── contact_sheet_generator.py  # Phase 1: Contact sheets
│   │   ├── llm_story_director.py       # Phase 2: AI story analysis
│   │   ├── audio_generator.py    # Phase 3: TTS generation
│   │   └── video_assembler.py    # Phase 4: Video assembly
│   ├── pipeline.py               # Main orchestrator
│   ├── api.py                    # FastAPI endpoints
│   ├── config.py                 # Configuration
│   └── main.py                   # Entry point
├── src/                          # React frontend
└── ...existing files...
```

## Setup

### Backend

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Install system dependencies:
- **poppler** (for pdf2image): `brew install poppler` or `sudo apt-get install poppler-utils`
- **ffmpeg** (for MoviePy): `brew install ffmpeg` or `sudo apt-get install ffmpeg`

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the backend:
```bash
python main.py
# or
uvicorn api:app --reload
```

### Frontend

1. Install dependencies:
```bash
npm install
```

2. Configure Supabase:
- Create a Supabase project
- Copy `.env` values from Supabase dashboard
- The database migration will be applied automatically

3. Run development server:
```bash
npm run dev
```

## Pipeline Phases

### Phase 1: PDF Processing & Pre-Extraction
- Convert PDF to high-res images
- Extract individual panels using OpenCV contour detection
- Generate contact sheets with labeled panel IDs

### Phase 2: LLM Story Director
- Send pages + contact sheet to VLM
- AI divides narrative into 4 parts
- Generates scripts (65-75 words each)
- Selects 5-7 panels per part

### Phase 3: Audio Generation
- Convert scripts to audio using edge-tts
- Calculate timing for each panel

### Phase 4: Video Assembly
- Apply Ken Burns (pan/zoom) effects
- Sync voiceover and panels
- Render 1080x1920 MP4 videos

## API Endpoints

- `POST /api/jobs/upload` - Upload PDF
- `POST /api/jobs/{job_id}/process` - Start processing
- `GET /api/jobs/{job_id}/status` - Get status
- `GET /api/jobs/{job_id}/videos/{part}` - Download video

## Notes

- Mock story analysis is used if LLM API keys are not configured
- Background music can be added via `background_music_path`
- All temporary files stored in `./temp_workspace/{job_id}/`
