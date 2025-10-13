"""
Voice Emotion Classification using Groq API
===========================================

Fast voice emotion analysis using Groq's Whisper + Llama models.
Provides audio transcription, emotion detection, and audio feature extraction.

Dependencies:
    pip install requests librosa numpy python-dotenv soundfile
"""

import os
import tempfile
from typing import Dict, Optional, Any, Tuple
import requests
import librosa
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class VoiceAnalyzer:
    """
    Voice Emotion Analyzer using Groq API
    
    Analyzes audio files to detect emotions using Groq's AI models.
    Combines Whisper for transcription and Llama for emotion detection.
    
    Usage:
        analyzer = VoiceAnalyzer(api_key="your_groq_api_key")
        result = analyzer.analyze_audio("audio.wav")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize Voice Analyzer
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY in environment)
            model: Groq model to use for emotion analysis
            progress_callback: Optional callback for progress updates
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required. Set it in environment or pass to constructor.")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.progress_callback = progress_callback or (lambda **kwargs: None)
        
        # Emotion mappings
        self.emotion_labels = {
            "anger": "Anger",
            "disgust": "Disgust",
            "fear": "Fear",
            "happiness": "Happiness",
            "neutral": "Neutral",
            "sadness": "Sadness",
            "surprise": "Surprise"
        }
        
        self.emotion_emoji = {
            "anger": "😠",
            "disgust": "🤢",
            "fear": "😨",
            "happiness": "😊",
            "neutral": "😐",
            "sadness": "😢",
            "surprise": "😮"
        }
        
        print(f"✅ Voice Analyzer initialized with Groq API ({self.model})")
    
    def _update_progress(self, stage: str, progress: float, message: str = ''):
        """Update progress callback"""
        try:
            self.progress_callback(stage=stage, progress=progress, message=message)
        except TypeError:
            try:
                self.progress_callback(stage, progress, message)
            except:
                pass
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file and return emotion classification
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            Dict containing:
            {
                "status": "success",
                "transcription": "...",
                "emotion": {...},
                "audio_features": {...},
                "metadata": {...}
            }
        """
        
        # Validate file
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "error": "Audio file not found",
                "audio_path": audio_path
            }
        
        try:
            # Step 1: Load audio and extract features
            self._update_progress('loading_audio', 0.1, 'Loading audio file...')
            speech, sample_rate = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=speech, sr=sample_rate)
            
            self._update_progress('extracting_features', 0.2, 'Extracting audio features...')
            audio_features = self._extract_audio_features(speech, sample_rate)
            
            # Step 2: Transcribe audio using Groq Whisper
            self._update_progress('transcribing', 0.3, 'Transcribing audio...')
            transcript, transcript_status = self._transcribe_audio_groq(audio_path)
            
            # Step 3: Analyze emotion from transcript
            if transcript and transcript_status == "success":
                self._update_progress('analyzing_emotion', 0.6, 'Analyzing voice emotion...')
                emotion_result = self._analyze_emotion_groq(transcript, audio_features)
            else:
                emotion_result = {
                    "dominant": "neutral",
                    "confidence": 0.5,
                    "all_emotions": {"neutral": 0.5},
                    "top_3": [{"emotion": "neutral", "confidence": 0.5}]
                }
            
            # Build response
            self._update_progress('finalizing', 0.9, 'Finalizing results...')
            response = {
                "status": "success",
                "transcription": transcript if transcript else "No transcription available",
                "transcription_status": transcript_status,
                "emotion": emotion_result,
                "audio_features": {
                    "duration_seconds": round(duration, 2),
                    "energy": audio_features["energy"],
                    "zero_crossing_rate": audio_features["zero_crossing_rate"],
                    "spectral_centroid": audio_features["spectral_centroid"],
                    "tempo": audio_features["tempo"],
                    "sample_rate": sample_rate
                },
                "metadata": {
                    "model": self.model,
                    "provider": "Groq API",
                    "file_name": os.path.basename(audio_path),
                    "has_transcription": bool(transcript)
                }
            }
            
            self._update_progress('complete', 1.0, 'Analysis complete!')
            return response
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "audio_path": audio_path
            }
    
    def _transcribe_audio_groq(self, audio_path: str) -> Tuple[str, str]:
        """
        Transcribe audio using Groq Whisper API
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (transcript, status)
        """
        try:
            # Convert to WAV if needed
            temp_wav = None
            if not audio_path.lower().endswith('.wav'):
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_wav.name
                temp_wav.close()
                
                # Convert using librosa
                audio, sr = librosa.load(audio_path, sr=16000)
                import soundfile as sf
                sf.write(temp_path, audio, sr)
                audio_path = temp_path
            
            # Call Groq Whisper API
            url = f"{self.base_url}/audio/transcriptions"
            
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
                
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120
                )
            
            # Cleanup temp file
            if temp_wav:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('text', '')
                return transcript, "success"
            else:
                return "", "error"
                
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return "", "error"
    
    def _analyze_emotion_groq(self, transcript: str, audio_features: Dict) -> Dict[str, Any]:
        """
        Analyze emotion from transcript using Groq API
        
        Args:
            transcript: Audio transcript
            audio_features: Extracted audio features
            
        Returns:
            Emotion analysis dictionary
        """
        
        # Build context-aware prompt
        energy_level = "high" if audio_features["energy"] > 0.02 else "low"
        tempo_info = f"tempo of {audio_features['tempo']} BPM"
        
        system_prompt = """You are an expert voice emotion analyst. Analyze the emotion in spoken text considering both the words and audio characteristics.

Classify the emotion into EXACTLY one of these categories:
- anger
- disgust
- fear
- happiness
- neutral
- sadness
- surprise

Respond in EXACTLY this JSON format:
{
  "emotion": "one of the 7 emotions above",
  "confidence": 0-100,
  "reasoning": "brief explanation",
  "secondary_emotions": {
    "emotion_name": confidence_score
  }
}"""

        user_prompt = f"""Analyze the emotion in this spoken text:

Text: "{transcript}"

Audio characteristics:
- Energy level: {energy_level}
- Speech tempo: {tempo_info}
- Spectral characteristics: {audio_features['spectral_centroid']:.0f} Hz

What emotion is being expressed?"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"}
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            import json
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            
            # Normalize emotion name
            emotion = parsed.get("emotion", "neutral").lower()
            confidence = float(parsed.get("confidence", 50)) / 100
            secondary = parsed.get("secondary_emotions", {})
            
            # Build all_emotions dict
            all_emotions = {emotion: confidence}
            for emo, score in secondary.items():
                emo_lower = emo.lower()
                if emo_lower in self.emotion_labels:
                    all_emotions[emo_lower] = float(score) / 100 if score > 1 else float(score)
            
            # Ensure all emotions are represented
            for emo_key in self.emotion_labels.keys():
                if emo_key not in all_emotions:
                    all_emotions[emo_key] = 0.01
            
            # Normalize to sum to 1
            total = sum(all_emotions.values())
            all_emotions = {k: v/total for k, v in all_emotions.items()}
            
            # Sort for top 3
            sorted_emotions = sorted(
                all_emotions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return {
                "dominant": emotion,
                "confidence": round(confidence, 4),
                "all_emotions": {k: round(v, 4) for k, v in all_emotions.items()},
                "top_3": [
                    {
                        "emotion": emo,
                        "confidence": round(score, 4)
                    }
                    for emo, score in sorted_emotions[:3]
                ]
            }
            
        except Exception as e:
            print(f"⚠️  Emotion analysis error: {e}")
            return {
                "dominant": "neutral",
                "confidence": 0.5,
                "all_emotions": {"neutral": 1.0},
                "top_3": [{"emotion": "neutral", "confidence": 1.0}]
            }
    
    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Extract audio features for analysis
        
        Args:
            audio: Audio signal array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of audio features
        """
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=audio)[0]
            energy = float(np.mean(rms))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral Centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate
            )[0]
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            
            return {
                "energy": round(energy, 4),
                "zero_crossing_rate": round(zcr_mean, 4),
                "spectral_centroid": round(spectral_centroid_mean, 2),
                "tempo": round(float(tempo), 2)
            }
            
        except Exception as e:
            print(f"⚠️  Feature extraction warning: {e}")
            return {
                "energy": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
                "tempo": 0.0
            }
    
    def get_emotion_interpretation(self, emotion: str, confidence: float) -> str:
        """
        Get human-readable interpretation of emotion
        
        Args:
            emotion: Emotion name
            confidence: Confidence score (0-1)
            
        Returns:
            Interpretation string
        """
        emoji = self.emotion_emoji.get(emotion.lower(), "")
        
        if confidence >= 0.8:
            level = "strong"
        elif confidence >= 0.6:
            level = "moderate"
        else:
            level = "slight"
        
        return f"{emoji} Detected {level} {emotion} emotion with {confidence*100:.1f}% confidence"


def get_voice_analyzer() -> VoiceAnalyzer:
    """Factory function to get VoiceAnalyzer instance"""
    return VoiceAnalyzer()
