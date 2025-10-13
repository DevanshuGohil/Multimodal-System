"""
Video Analysis Module - FIXED VERSION
=====================================

A multimodal video analysis module that combines:
- Speech-to-Text using Groq's Whisper
- Emotion detection using Face++
- Text sentiment analysis using Hugging Face
"""

import os
import json
import subprocess
import hashlib
import tempfile
import base64  # FIX: Added missing import
from datetime import datetime  # FIX: Correct import
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'cache_enabled': True,
    'cache_dir': './cache',
    'max_frames': 10,
    'extract_fps': 1,
    'api_timeout': 120,
}


def is_youtube_url(url: str) -> bool:
    """Check if input is a YouTube URL"""
    youtube_domains = [
        'youtube.com',
        'youtu.be',
        'www.youtube.com',
        'm.youtube.com'
    ]
    return any(domain in url.lower() for domain in youtube_domains)


class GroqSTTClient:
    """Groq Speech-to-Text client using Whisper API"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
            
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"  # FIX: Correct URL
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Groq's Whisper API"""
        try:
            url = f"{self.base_url}/audio/transcriptions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(audio_path), audio_file, 'audio/wav')
                }
                data = {
                    'model': 'whisper-large-v3-turbo',
                    'response_format': 'json',
                    'language': 'en',
                    'temperature': '0'
                }
                
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=CONFIG['api_timeout']
                )
                
                response.raise_for_status()
                return response.json().get('text', '')
                
        except Exception as e:
            print(f"Error in STT: {str(e)}")
            return ""


class FacePlusPlusClient:
    """Face++ client for emotion detection"""
    
    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError("FACEPP_API_KEY and FACEPP_API_SECRET are required")
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
    
    def detect_emotion_from_image(self, image_path: str) -> Dict:  # FIX: Return Dict, not Tuple
        """Detect emotion from an image using Face++ API"""
        try:
            with open(image_path, 'rb') as f:
                files = {'image_file': f}
                data = {
                    'api_key': self.api_key,
                    'api_secret': self.api_secret,
                    'return_attributes': 'emotion'
                }
                
                response = requests.post(
                    self.base_url,
                    files=files,
                    data=data,
                    timeout=CONFIG['api_timeout']
                )
                
                response.raise_for_status()
                result = response.json()
            
            # FIX: Return consistent Dict format
            if 'faces' in result and len(result['faces']) > 0:
                emotions = result['faces'][0]['attributes']['emotion']
                normalized = {k: float(v) / 100.0 for k, v in emotions.items()}
                dominant = max(normalized, key=normalized.get)
                
                return {
                    'dominant_emotion': dominant,
                    'confidence': normalized[dominant],
                    'all_emotions': normalized
                }
            
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {}
            }
            
        except Exception as e:
            print(f"Error in Face++ API: {str(e)}")
            return {
                'dominant_emotion': 'error',
                'confidence': 0.0,
                'all_emotions': {}
            }


class HuggingFaceTextClient:
    """Client for Hugging Face text emotion analysis"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("HUGGING_FACE_API_KEY is required")
            
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.model = "j-hartmann/emotion-english-distilroberta-base"
    
    def detect_emotion_from_text(self, text: str) -> Dict:  # FIX: Return Dict, not Tuple
        """Detect emotion from text using Hugging Face API"""
        try:
            if not text or len(text.strip()) < 10:
                return {'label': 'neutral', 'score': 1.0, 'all_emotions': []}
            
            url = f"{self.base_url}/{self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": text[:512]},
                timeout=CONFIG['api_timeout']
            )
            
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]
                
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score'],
                    'all_emotions': result[:3]
                }
            
            return {'label': 'neutral', 'score': 1.0, 'all_emotions': []}
            
        except Exception as e:
            print(f"Error in Hugging Face API: {str(e)}")
            return {'label': 'error', 'score': 0.0, 'all_emotions': []}


class VideoAnalyzer:
    """
    Video Analysis Model
    Analyzes videos for emotion, sincerity, and authenticity
    
    Usage:
        analyzer = VideoAnalyzer(
            groq_api_key="your_key",
            facepp_api_key="your_key",
            facepp_api_secret="your_secret",
            huggingface_api_key="your_key"
        )
        result = analyzer.analyze_video("video.mp4")
    """
    
    def __init__(
        self,
        groq_api_key: str = None,
        facepp_api_key: str = None,
        facepp_api_secret: str = None,
        huggingface_api_key: str = None,
        max_frames: int = 10,
        extract_fps: int = 1
    ):
        """Initialize Video Analyzer"""
        
        # Get API keys from env or parameters
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.facepp_api_key = facepp_api_key or os.getenv("FACEPP_API_KEY")
        self.facepp_api_secret = facepp_api_secret or os.getenv("FACEPP_API_SECRET")
        self.huggingface_api_key = huggingface_api_key or os.getenv("HUGGING_FACE_API_KEY")
        
        self.max_frames = max_frames
        self.extract_fps = extract_fps
        
        # Validate API keys
        self._validate_api_keys()
        
        # FIX: Initialize client instances
        self.groq = GroqSTTClient(self.groq_api_key)
        self.facepp = FacePlusPlusClient(self.facepp_api_key, self.facepp_api_secret)
        self.hf_text = HuggingFaceTextClient(self.huggingface_api_key)
        
        # Create cache directory
        if CONFIG['cache_enabled']:
            os.makedirs(CONFIG['cache_dir'], exist_ok=True)
    
    def _validate_api_keys(self):
        """Validate that all required API keys are provided"""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        if not self.facepp_api_key or not self.facepp_api_secret:
            raise ValueError("FACEPP_API_KEY and FACEPP_API_SECRET are required")
        if not self.huggingface_api_key:
            raise ValueError("HUGGING_FACE_API_KEY is required")
    
    def analyze_video(self, video_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze video and return detailed JSON response
        
        Args:
            video_path: Path to video file or YouTube URL
            use_cache: Whether to use cached results if available
            
        Returns:
            Dict containing complete analysis results
        """
        
        # Check cache first
        if use_cache and CONFIG['cache_enabled']:
            cache_path = self._get_cache_path(video_path)
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)
        
        # Handle YouTube URLs
        downloaded_file = None
        if is_youtube_url(video_path):
            video_path = self._download_youtube_video(video_path)
            downloaded_file = video_path
        
        # Validate video file
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "error": "Video file not found",
                "video_path": video_path
            }
        
        temp_files = []
        
        try:
            # Get video metadata
            video_info = self._get_video_info(video_path)
            
            # Extract audio
            audio_path = self._extract_audio(video_path)
            temp_files.append(audio_path)
            
            # Transcribe
            transcript = self.groq.transcribe(audio_path)
            
            # Extract frames
            frames = self._extract_frames(video_path)
            temp_files.extend(frames)
            
            # Analyze face emotion
            if frames:
                face_result = self.facepp.detect_emotion_from_image(frames[0])
            else:
                face_result = {
                    'dominant_emotion': 'neutral',
                    'confidence': 0.0,
                    'all_emotions': {}
                }
            
            # Analyze text emotion
            text_result = self.hf_text.detect_emotion_from_text(transcript)
            
            # Extract audio features
            audio_features = self._extract_audio_features(audio_path)
            
            # Calculate sincerity
            sincerity = self._calculate_sincerity(
                face_result['dominant_emotion'],
                text_result['label'],
                transcript,
                audio_features,
                face_result['confidence']
            )
            
            # Build response
            response = {
                "status": "success",
                "video_info": {
                    "filename": os.path.basename(video_path),
                    "duration_seconds": video_info['duration'],
                    "fps": video_info['fps'],
                    "resolution": f"{video_info['width']}x{video_info['height']}",
                    "frames_analyzed": len(frames)
                },
                "transcript": {
                    "text": transcript,
                    "word_count": len(transcript.split()),
                    "character_count": len(transcript)
                },
                "emotions": {
                    "face": {
                        "dominant": face_result['dominant_emotion'],
                        "confidence": round(face_result['confidence'], 4),
                        "all_emotions": {
                            k: round(v, 4) 
                            for k, v in face_result['all_emotions'].items()
                        }
                    },
                    "text": {
                        "dominant": text_result['label'],
                        "confidence": round(text_result['score'], 4),
                        "top_emotions": [
                            {
                                "emotion": e['label'],
                                "confidence": round(e['score'], 4)
                            }
                            for e in text_result.get('all_emotions', [])[:3]
                        ]
                    }
                },
                "sincerity": {
                    "overall_score": round(sincerity['sincerity_score'], 4),
                    "emotion_congruence": round(sincerity['emotion_congruence'], 4),
                    "speech_quality": round(sincerity['speech_quality'], 4),
                    "voice_consistency": round(sincerity['voice_consistency'], 4),
                    "filler_word_count": sincerity['filler_word_count'],
                    "filler_ratio": round(sincerity['filler_ratio'], 4),
                    "interpretation": self._interpret_sincerity(sincerity['sincerity_score'])
                },
                "audio_features": {
                    "mean_volume_db": audio_features['mean_volume_db'],
                    "max_volume_db": audio_features['max_volume_db'],
                    "rms_energy": audio_features['rms_energy'],
                    "energy_level": self._interpret_energy(audio_features['rms_energy'])
                },
                "metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "model_versions": {
                        "speech_to_text": "whisper-large-v3-turbo",
                        "face_emotion": "Face++ v3",
                        "text_sentiment": "emotion-english-distilroberta-base"
                    }
                }
            }
            
            # Cache results
            if use_cache and CONFIG['cache_enabled']:
                cache_path = self._get_cache_path(video_path)
                with open(cache_path, 'w') as f:
                    json.dump(response, f)
            
            return response
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "video_path": video_path
            }
        
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            
            # Cleanup downloaded YouTube video
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.remove(downloaded_file)
                except:
                    pass
    
    # Alias for backward compatibility
    def analyze(self, video_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """Alias for analyze_video"""
        return self.analyze_video(video_path, use_cache)
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video"""
        audio_path = os.path.join(
            tempfile.gettempdir(),
            f"audio_{os.getpid()}_{hash(video_path)}.wav"
        )
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    
    def _extract_frames(self, video_path: str) -> List[str]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.extract_fps)
        
        frames = []
        count = 0
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                frame_path = os.path.join(
                    tempfile.gettempdir(),
                    f"frame_{os.getpid()}_{hash(video_path)}_{count}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                frame_count += 1
            
            count += 1
        
        cap.release()
        return frames
    
    def _extract_audio_features(self, audio_path: str) -> Dict:
        """Extract audio features using ffmpeg"""
        try:
            result = subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-af', 'volumedetect',
                '-f', 'null', '/dev/null'
            ], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            
            mean_volume = 0.0
            max_volume = 0.0
            
            output = result.stderr + result.stdout
            
            for line in output.split('\n'):
                if 'mean_volume:' in line:
                    try:
                        mean_volume = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'max_volume:' in line:
                    try:
                        max_volume = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
            
            rms_energy = max(0, min(1.0, abs(mean_volume) / 50.0))
            
            return {
                'mean_volume_db': round(mean_volume, 2),
                'max_volume_db': round(max_volume, 2),
                'rms_energy': round(rms_energy, 4)
            }
        
        except Exception as e:
            return {
                'mean_volume_db': 0,
                'max_volume_db': 0,
                'rms_energy': 0.5
            }
    
    def _calculate_sincerity(
        self,
        face_emotion: str,
        text_emotion: str,
        transcript: str,
        audio_features: Dict,
        face_confidence: float
    ) -> Dict:
        """Calculate multimodal sincerity score"""
        
        # Emotion congruence
        emotion_match = 1.0 if face_emotion.lower() == text_emotion.lower() else 0.5
        
        # Emotion synonyms
        emotion_synonyms = {
            'joy': ['happy', 'happiness'],
            'happiness': ['joy', 'happy'],
            'sadness': ['sad'],
            'sad': ['sadness'],
            'anger': ['angry'],
            'angry': ['anger'],
        }
        
        if face_emotion.lower() in emotion_synonyms:
            if text_emotion.lower() in emotion_synonyms[face_emotion.lower()]:
                emotion_match = 0.8
        
        # Weight by face confidence
        emotion_match = emotion_match * (0.5 + 0.5 * face_confidence)
        
        # Speech quality
        filler_words = [
            'um', 'uh', 'like', 'you know', 'basically',
            'actually', 'literally', 'sort of', 'kind of'
        ]
        transcript_lower = transcript.lower()
        filler_count = sum(transcript_lower.count(f' {word} ') for word in filler_words)
        
        word_count = len(transcript.split())
        filler_ratio = filler_count / max(word_count, 1)
        speech_quality = max(0, 1.0 - (filler_ratio * 10))
        
        # Voice consistency
        voice_consistency = audio_features.get('rms_energy', 0.5)
        
        # Overall sincerity
        sincerity = (
            0.5 * emotion_match +
            0.3 * speech_quality +
            0.2 * voice_consistency
        )
        
        return {
            'sincerity_score': sincerity,
            'emotion_congruence': emotion_match,
            'speech_quality': speech_quality,
            'voice_consistency': voice_consistency,
            'filler_word_count': filler_count,
            'filler_ratio': filler_ratio
        }
    
    def _interpret_sincerity(self, score: float) -> str:
        """Interpret sincerity score"""
        if score >= 0.8:
            return "High authenticity - strong congruence"
        elif score >= 0.6:
            return "Moderate authenticity - some mixed signals"
        elif score >= 0.4:
            return "Low authenticity - significant inconsistencies"
        else:
            return "Very low authenticity - major red flags"
    
    def _interpret_energy(self, rms: float) -> str:
        """Interpret audio energy level"""
        if rms > 0.7:
            return "high"
        elif rms > 0.3:
            return "medium"
        else:
            return "low"
    
    def _get_cache_path(self, video_path: str) -> str:
        """Get cache file path"""
        video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
        return os.path.join(CONFIG['cache_dir'], f"{video_hash}.json")
    
    def _download_youtube_video(self, url: str) -> str:
        """Download YouTube video using yt-dlp"""
        try:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"youtube_{os.getpid()}.mp4"
            )
            
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '-o', output_path,
                '--quiet',
                url
            ]
            
            subprocess.run(cmd, check=True)
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Error downloading YouTube video: {str(e)}")


def get_video_analyzer() -> VideoAnalyzer:
    """Factory function to get a VideoAnalyzer instance"""
    return VideoAnalyzer()
