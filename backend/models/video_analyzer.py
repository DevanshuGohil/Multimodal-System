import cv2
import librosa
import numpy as np
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from moviepy import VideoFileClip
from deepface import DeepFace
import whisper
from transformers import pipeline

class VideoAnalyzer:
    """
    Multi-modal video analysis combining facial expressions, voice, and speech content
    """
    
    def __init__(self):
        print("Initializing video analysis models...")
        # Initialize Whisper for speech-to-text
        self.whisper_model = whisper.load_model("base")
        
        # Initialize text emotion classifier
        self.text_emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        print("Video analysis models initialized!")
    
    def extract_audio(self, video_path: str, audio_path: Optional[str] = None) -> str:
        """Extract audio from video file"""
        if audio_path is None:
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "temp_audio.wav")
            
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return audio_path
    
    def extract_frames(self, video_path: str, frame_rate: int = 1) -> List[np.ndarray]:
        """Extract frames from video at specified rate"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate) if fps > frame_rate else 1
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frames.append(frame)
            count += 1
        
        cap.release()
        return frames
    
    def analyze_faces(self, frames: List[np.ndarray]) -> Dict:
        """Analyze facial attributes in video frames"""
        emotion_results = []
        age_results = []
        gender_results = []
        race_results = []
        
        for frame in frames:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion', 'age', 'gender', 'race'],
                    enforce_detection=False,
                    silent=True
                )
                if result and len(result) > 0:
                    emotion_results.append(result[0]['emotion'])
                    age_results.append(result[0]['age'])
                    gender_results.append(result[0]['dominant_gender'])
                    race_results.append(result[0]['dominant_race'])
            except Exception:
                continue
        
        return {
            'emotions': emotion_results,
            'ages': age_results,
            'genders': gender_results,
            'races': race_results
        }
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """Extract audio features using librosa"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        return {
            'mfcc_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist(),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
            'rms': float(np.mean(librosa.feature.rms(y=y))),
            'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0])
        }
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe speech to text using Whisper"""
        result = self.whisper_model.transcribe(
            audio_path,
            fp16=False,  # Disable FP16 for CPU compatibility
            language='en',
            verbose=False
        )
        return result["text"]
    
    def analyze_text_emotion(self, text: str) -> List[Dict]:
        """Analyze emotion from transcribed text"""
        if not text or len(text.strip()) < 10:
            return [{"label": "neutral", "score": 1.0}]
        
        # Limit to first 512 tokens to avoid max length issues
        return self.text_emotion_classifier(text[:512])[0]
    
    def analyze_video(self, video_path: str) -> Dict:
        """Main method to analyze video"""
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract and process audio
            audio_path = self.extract_audio(video_path, os.path.join(temp_dir, "audio.wav"))
            audio_features = self.extract_audio_features(audio_path)
            transcript = self.transcribe_audio(audio_path)
            
            # Extract and analyze frames
            frames = self.extract_frames(video_path, frame_rate=1)
            face_analysis = self.analyze_faces(frames)
            
            # Analyze text emotion
            text_emotions = self.analyze_text_emotion(transcript)
            
            # Format results
            return {
                'transcript': transcript,
                'facial_analysis': face_analysis,
                'audio_features': audio_features,
                'text_emotions': text_emotions,
                'num_frames_analyzed': len(frames),
                'num_faces_detected': len(face_analysis['emotions'])
            }
