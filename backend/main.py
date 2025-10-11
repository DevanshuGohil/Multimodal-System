from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
import os
import tempfile
import shutil
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

# In-memory storage for progress updates (in production, use a proper message broker like Redis)
progress_store = {}

def update_progress(analysis_id: str, status: str, message: str, percentage: int, step: int):
    """Update progress for a specific analysis"""
    progress_store[analysis_id] = {
        'status': status,
        'message': message,
        'percentage': percentage,
        'step': step
    }

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_summarizer import TextSummarizer
from models.face_analyzer import FaceAnalyzer
from models.voice_classifier import VoiceClassifier
from models.video_analyzer import VideoAnalyzer

app = FastAPI(
    title="AI Multi-Modal Analysis API",
    description="Dynamic AI-powered analysis for text, images, voice, and video using transformer models",
    version="1.0.0"
)

# CORS middleware - Allow frontend origins
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://*.railway.app",
    "https://*.vercel.app",
    "https://*.netlify.app",
]

# Get custom origins from environment variable if set
custom_origins = os.environ.get("ALLOWED_ORIGINS", "")
if custom_origins:
    allowed_origins.extend(custom_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.railway\.app"
)

# Initialize models (lazy loading)
sentiment_analyzer = None
text_summarizer = None
face_analyzer = None
voice_classifier = None
video_analyzer = None


def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentAnalyzer()
    return sentiment_analyzer


def get_text_summarizer():
    global text_summarizer
    if text_summarizer is None:
        text_summarizer = TextSummarizer()
    return text_summarizer


def get_face_analyzer():
    global face_analyzer
    if face_analyzer is None:
        face_analyzer = FaceAnalyzer()
    return face_analyzer


def get_voice_classifier():
    global voice_classifier
    if voice_classifier is None:
        voice_classifier = VoiceClassifier()
    return voice_classifier


def get_video_analyzer():
    global video_analyzer
    if video_analyzer is None:
        video_analyzer = VideoAnalyzer()
    return video_analyzer


@app.get("/")
async def root():
    return {
        "message": "AI Multi-Modal Analysis API",
        "endpoints": {
            "sentiment": "/api/sentiment",
            "summarize": "/api/summarize",
            "face-analysis": "/api/face-analysis",
            "voice-analysis": "/api/voice-analysis",
            "video-analysis": "/api/video-analysis"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/sentiment")
async def analyze_sentiment(text: str = Form(...)):
    """
    Analyze sentiment of text using transformer-based model (DistilBERT)
    """
    try:
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze(text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_text(
    text: str = Form(...),
    max_length: Optional[int] = Form(130),
    min_length: Optional[int] = Form(30)
):
    """
    Summarize text using transformer-based model (BART)
    """
    try:
        summarizer = get_text_summarizer()
        result = summarizer.summarize(text, max_length=max_length, min_length=min_length)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face-analysis")
async def analyze_face(file: UploadFile = File(...)):
    """
    Analyze face in image for age, gender, emotion, and facial attributes
    """
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        analyzer = get_face_analyzer()
        result = analyzer.analyze(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")


@app.post("/api/voice-analysis")
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze voice characteristics from audio file
    """
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        analyzer = get_voice_classifier()
        result = analyzer.analyze(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

async def analyze_video_file(file: UploadFile, analysis_id: str):
    """Process video file and update progress"""
    temp_file_path = None
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Get the video analyzer
        analyzer = get_video_analyzer()

        # Simulate processing steps (replace with actual processing)
        steps = [
            ("processing", "Processing video...", 20, 1),
            ("processing_audio", "Processing audio...", 40, 2),
            ("analyzing_faces", "Analyzing facial expressions...", 60, 3),
            ("finalizing", "Finalizing results...", 80, 4),
            ("completed", "Analysis complete!", 100, 5)
        ]

        for status, message, percentage, step in steps:
            update_progress(analysis_id, status, message, percentage, step)
            await asyncio.sleep(2)  # Simulate processing time

        # Process the video
        result = analyzer.analyze_video(temp_file_path)
        
        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        # Store the final result
        if analysis_id in progress_store:
            progress_store[analysis_id]['result'] = result
            
        return result

    except Exception as e:
        update_progress(analysis_id, "error", f"Analysis failed: {str(e)}", 100, 5)
        raise
    finally:
        # Clean up the temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")

@app.post("/api/video-analysis")
async def analyze_video(file: UploadFile = File(...)):
    """
    Start video analysis and return an analysis ID
    """
    analysis_id = str(uuid.uuid4())
    
    # Initialize progress
    update_progress(analysis_id, "queued", "Waiting to start analysis...", 0, 0)
    
    # Start the analysis in the background
    asyncio.create_task(analyze_video_file(file, analysis_id))
    
    return {
        "status": "started",
        "analysis_id": analysis_id,
        "message": "Analysis started. Use the analysis_id to track progress.",
        "progress_url": f"/api/video-analysis/progress/{analysis_id}"
    }

@app.get("/api/video-analysis/progress/{analysis_id}")
async def video_progress(analysis_id: str, request: Request):
    """SSE endpoint for video analysis progress"""
    return StreamingResponse(
        event_generator(analysis_id),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable buffering for nginx
        }
    )

async def event_generator(analysis_id: str) -> AsyncGenerator[str, None]:
    """Generate server-sent events for progress updates"""
    last_progress = {}
    
    try:
        while True:
            # Get current progress
            current_progress = progress_store.get(analysis_id, {
                'status': 'queued',
                'message': 'Waiting to start analysis...',
                'percentage': 0,
                'step': 0,
                'total_steps': 5
            })
            
            # Only send updates if something changed
            if current_progress != last_progress:
                last_progress = current_progress.copy()
                yield f"data: {json.dumps(current_progress)}\n\n"
                
                # If analysis is complete, include the result in the final message
                if current_progress.get('status') == 'completed' and 'result' in progress_store.get(analysis_id, {}):
                    final_result = {
                        'status': 'completed',
                        'message': 'Analysis complete!',
                        'percentage': 100,
                        'step': 5,
                        'total_steps': 5,
                        'result': progress_store[analysis_id]['result']
                    }
                    yield f"data: {json.dumps(final_result)}\n\n"
                    break
                    
            await asyncio.sleep(1)  # Check for updates every second
            
    except asyncio.CancelledError:
        print(f"Client disconnected from progress stream for {analysis_id}")
    except Exception as e:
        print(f"Error in event generator: {str(e)}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
