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
    print("Starting up...")
    sentiment_analyzer = SentimentAnalyzer()
    text_summarizer = TextSummarizer()
    voice_classifier = VoiceClassifier()
    video_analyzer = VideoAnalyzer(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        facepp_api_key=os.getenv("FACEPP_API_KEY"),
        facepp_api_secret=os.getenv("FACEPP_API_SECRET"),
        huggingface_api_key=os.getenv("HUGGING_FACE_API_KEY")
    )
    face_analyzer = FaceAnalyzer()
    
    yield  # This is where the application runs
    
    # Shutdown: Clean up resources if needed
    print("Shutting down...")

# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="AI Multi-Modal Analysis API",
    description="Dynamic AI-powered analysis for text, images, voice, and video using transformer models",
    version="1.0.0",
    lifespan=lifespan
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

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: Initialize models
    print("Starting up...")
    get_sentiment_analyzer()
    get_text_summarizer()
    get_voice_classifier()
    get_video_analyzer()
    get_face_analyzer()
    
    yield  # This is where the application runs
    
    # Shutdown: Clean up resources if needed
    print("Shutting down...")


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
    if not hasattr(get_voice_classifier, 'instance'):
        get_voice_classifier.instance = VoiceClassifier()
    return get_voice_classifier.instance


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

@app.get("/")
async def root():
    return {
        "message": "AI Multi-Modal Analysis API",
        "endpoints": {
            "sentiment": "/api/sentiment",
            "summarize": "/api/summarize",
            "voice-analysis": "/api/voice-analysis",
            "video-analysis": "/api/video-analysis",
            "face-analysis": "/api/face-analysis"
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
        result = analyzer.analyze_audio(temp_file_path)
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

async def analyze_video_file(file: UploadFile):
    """Process video file and return analysis results"""
    temp_file_path = None
    try:
        print(f"Starting video analysis for file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file type. Supported formats: .mp4, .mov, .avi, .mkv")

        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            print(f"Created temporary file: {temp_file.name}")
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            print(f"Copied video data to {temp_file_path}, size: {os.path.getsize(temp_file_path)} bytes")

        # Get the video analyzer and process the video
        print("Initializing video analyzer...")
        try:
            analyzer = get_video_analyzer()
            print("Analyzing video...")
            result = analyzer.analyze_video(temp_file_path, use_cache=False)  # Disable cache for debugging
            print("Video analysis completed successfully")
            
            # Check if the analysis was successful
            if result.get('status') == 'error':
                error_msg = result.get('error', 'Video analysis failed')
                print(f"Analysis error: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
                
            return result
            
        except Exception as analyzer_error:
            print(f"Error during video analysis: {str(analyzer_error)}")
            if hasattr(analyzer_error, 'response') and hasattr(analyzer_error.response, 'text'):
                print(f"API Error response: {analyzer_error.response.text}")
            raise

    except HTTPException as http_error:
        print(f"HTTP Error: {http_error.detail}")
        raise
        
    except Exception as e:
        error_msg = f"Video analysis failed: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    finally:
        # Clean up the temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")

@app.post("/api/video-analysis")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze video and return results
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: MP4, MOV, AVI, MKV"
            )
            
        # Process the video and return results directly
        result = await analyze_video_file(file)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
