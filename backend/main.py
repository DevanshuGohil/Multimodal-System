from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import os
import tempfile
import shutil

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.sentiment_analyzer import SentimentAnalyzer
from models.text_summarizer import TextSummarizer
from models.face_analyzer import FaceAnalyzer
from models.voice_classifier import VoiceClassifier

app = FastAPI(
    title="AI Multi-Modal Analysis API",
    description="Dynamic AI-powered analysis for text, images, and voice using transformer models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
sentiment_analyzer = None
text_summarizer = None
face_analyzer = None
voice_classifier = None


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


@app.get("/")
async def root():
    return {
        "message": "AI Multi-Modal Analysis API",
        "endpoints": {
            "sentiment": "/api/sentiment",
            "summarize": "/api/summarize",
            "face-analysis": "/api/face-analysis",
            "voice-analysis": "/api/voice-analysis"
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
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            analyzer = get_face_analyzer()
            result = analyzer.analyze(tmp_path)
            return JSONResponse(content=result)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice-analysis")
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze voice tone and emotion from audio file using Wav2Vec2
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            classifier = get_voice_classifier()
            result = classifier.classify(tmp_path)
            return JSONResponse(content=result)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
