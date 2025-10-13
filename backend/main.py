import os
import tempfile
import shutil
import uvicorn
import numpy as np
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_summarizer import TextSummarizer
from models.voice_classifier import VoiceAnalyzer as VoiceClassifier
from models.video_analyzer import VideoAnalyzer
from models.face_analyzer import FaceAnalyzer

# Global variables
sentiment_analyzer = None
text_summarizer = None
voice_classifier = None
video_analyzer = None
face_analyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handle application lifespan events (startup and shutdown).
    """
    global sentiment_analyzer, text_summarizer, voice_classifier, video_analyzer, face_analyzer
    
    # Startup: Initialize models
    print("🚀 Starting up application...")
    print("📦 Loading models...")
    
    try:
        sentiment_analyzer = SentimentAnalyzer()
        print("✅ Sentiment Analyzer loaded")
        
        text_summarizer = TextSummarizer()
        print("✅ Text Summarizer loaded")
        
        voice_classifier = VoiceClassifier()
        print("✅ Voice Classifier loaded")
        
        video_analyzer = VideoAnalyzer(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            facepp_api_key=os.getenv("FACEPP_API_KEY"),
            facepp_api_secret=os.getenv("FACEPP_API_SECRET"),
            huggingface_api_key=os.getenv("HUGGING_FACE_API_KEY")
        )
        print("✅ Video Analyzer loaded")
        
        face_analyzer = FaceAnalyzer()
        print("✅ Face Analyzer loaded")
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown: Clean up resources
    print("🛑 Shutting down application...")


# ============================================
# CREATE FASTAPI APP - MUST BE BEFORE MIDDLEWARE
# ============================================

app = FastAPI(
    title="AI Multi-Modal Analysis API",
    description="Sentiment, Voice, Video, and Face Analysis powered by AI",
    version="1.0.0",
    lifespan=lifespan,  # CRITICAL: Add lifespan parameter
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================
# CORS CONFIGURATION - AFTER APP CREATION
# ============================================

# Define allowed origins explicitly (wildcards don't work in CORS)
allowed_origins = [
    # Local development
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "https://localhost:3000",
    "https://localhost:5173",
    
    # Production domains
    "https://multimodal-system.vercel.app",
    "https://multimodal-system.onrender.com",
    
    # Add your specific Vercel preview URLs here as you get them
    # Example: "https://multimodal-system-git-main-yourname.vercel.app",
]

# Get custom origins from environment variable
custom_origins = os.environ.get("ALLOWED_ORIGINS", "")
if custom_origins:
    allowed_origins.extend([origin.strip() for origin in custom_origins.split(",")])

# Remove empty strings and duplicates
allowed_origins = list(set(filter(None, allowed_origins)))

print(f"🌐 CORS enabled for origins: {allowed_origins}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)


# ============================================
# HELPER FUNCTIONS
# ============================================

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
        video_analyzer = VideoAnalyzer(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            facepp_api_key=os.getenv("FACEPP_API_KEY"),
            facepp_api_secret=os.getenv("FACEPP_API_SECRET"),
            huggingface_api_key=os.getenv("HUGGING_FACE_API_KEY")
        )
    return video_analyzer


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
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


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Multi-Modal Analysis API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "sentiment": "/api/sentiment",
            "summarize": "/api/summarize",
            "voice-analysis": "/api/voice-analysis",
            "video-analysis": "/api/video-analysis",
            "face-analysis": "/api/face-analysis",
            "docs": "/docs",
            "health": "/health"
        },
        "cors_enabled": True,
        "allowed_origins": len(allowed_origins)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cors_enabled": True,
        "models_loaded": {
            "sentiment": sentiment_analyzer is not None,
            "summarizer": text_summarizer is not None,
            "voice": voice_classifier is not None,
            "video": video_analyzer is not None,
            "face": face_analyzer is not None
        }
    }


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle preflight OPTIONS requests"""
    return JSONResponse(content={"status": "ok"})


@app.post("/api/sentiment")
async def analyze_sentiment(text: str = Form(...)):
    """
    Analyze sentiment of text using transformer-based model
    """
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze(text)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
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
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        summarizer = get_text_summarizer()
        result = summarizer.summarize(text, max_length=max_length, min_length=min_length)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face-analysis")
async def analyze_face(file: UploadFile = File(...)):
    """
    Analyze face in image for age, gender, emotion, and facial attributes
    """
    temp_file_path = None
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: JPG, JPEG, PNG"
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Analyze
        analyzer = get_face_analyzer()
        result = analyzer.analyze(temp_file_path)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in face analysis: {e}")
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
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: WAV, MP3, M4A, FLAC"
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Analyze
        analyzer = get_voice_classifier()
        result = analyzer.analyze_audio(temp_file_path)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in voice analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")


@app.post("/api/video-analysis")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze video for emotions, speech, and audio features
    """
    temp_file_path = None
    try:
        print(f"📹 Starting video analysis for: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: MP4, MOV, AVI, MKV"
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            file_size = os.path.getsize(temp_file_path)
            print(f"✅ Video saved: {file_size / 1024 / 1024:.2f} MB")
        
        # Analyze
        print("🔍 Analyzing video...")
        analyzer = get_video_analyzer()
        result = analyzer.analyze_video(temp_file_path, use_cache=False)
        
        # Check for errors
        if result.get('status') == 'error':
            error_msg = result.get('error', 'Video analysis failed')
            print(f"❌ Analysis error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        print("✅ Video analysis completed successfully")
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in video analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"🧹 Cleaned up temporary file")
            except Exception as e:
                print(f"⚠️  Could not delete temporary file: {e}")


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
