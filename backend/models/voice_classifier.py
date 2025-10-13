"""
Voice Emotion Classification Module
===================================

A production-ready voice emotion classifier using Wav2Vec2 transformer model.
Provides emotion detection, audio feature extraction, and detailed analysis.

Dependencies:
    pip install torch torchaudio transformers librosa numpy
"""

import os
import tempfile
from typing import Dict, Optional, Any, Tuple
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment


class VoiceAnalyzer:
    """
    Voice Emotion Analyzer
    
    Analyzes audio files to detect emotions using Wav2Vec2 transformer model.
    Returns detailed emotion analysis with confidence scores and audio features.
    
    Usage:
        analyzer = VoiceAnalyzer()
        result = analyzer.analyze_audio("audio.wav")
    """
    
    def __init__(
        self,
        model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ):
        """Initialize Voice Analyzer
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to run the model on ('cuda' or 'cpu')
            progress_callback: Optional callback function for progress updates
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback or (lambda **kwargs: None)
        
        # Emotion mappings
        self.emotion_labels = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "happiness",
            4: "neutral",
            5: "sadness",
            6: "surprise"
        }
        
        # Initialize models and processors
        self.model = None
        self.feature_extractor = None
        self.speech_recognizer = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all required models and processors"""
        try:
            self._update_progress('loading_models', 0.2, 'Loading voice analysis model...')
            
            # Load emotion classification model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.emotion_labels),
                output_hidden_states=False,
                output_attentions=False,
                gradient_checkpointing=True
            ).to(self.device)
            
            # Initialize speech recognizer
            self.speech_recognizer = sr.Recognizer()
            self.speech_recognizer.dynamic_energy_threshold = True
            self.speech_recognizer.pause_threshold = 0.8
            self.speech_recognizer.phrase_threshold = 0.3
            self.speech_recognizer.non_speaking_duration = 0.5
            
            # Test speech recognizer initialization
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav') as test_file:
                    # Create a silent audio file for testing
                    AudioSegment.silent(duration=100).export(test_file.name, format='wav')
                    with sr.AudioFile(test_file.name) as source:
                        self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("✅ Speech recognizer initialized successfully!")
            except Exception as e:
                print(f"⚠️  Speech recognizer test failed: {e}")
                print("⚠️  Transcription may not work properly")
            
            self._update_progress('loading_models', 1.0, 'Models loaded successfully!')
            print("✅ Voice analysis model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("⚠️  Some features may be limited")
            if not hasattr(self, 'model'):
                self.model = None
    
    def _update_progress(self, stage: str, progress: float, message: str = '') -> None:
        """Helper method to update progress
        
        Args:
            stage: Current processing stage
            progress: Progress value (0.0 to 1.0)
            message: Optional status message
        """
        if self.progress_callback is not None:
            try:
                # Try calling with keyword arguments first
                self.progress_callback(stage=stage, progress=progress, message=message)
            except TypeError:
                try:
                    # Fall back to positional arguments
                    self.progress_callback(stage, progress, message)
                except Exception as e:
                    print(f"⚠️  Error in progress callback: {e}")

    def _transcribe_audio(self, audio_path: str) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio to text using speech recognition"""
        tmp_wav_path = None
        try:
            # Initialize speech recognizer if not already done
            if not hasattr(self, 'speech_recognizer') or self.speech_recognizer is None:
                self.speech_recognizer = sr.Recognizer()
            
            # Convert audio to WAV if needed
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
            
            # Convert to WAV format if needed
            try:
                if not audio_path.lower().endswith('.wav'):
                    audio = AudioSegment.from_file(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(tmp_wav_path, format='wav')
                    audio_path = tmp_wav_path
            except Exception as e:
                print(f"⚠️  Audio conversion error: {e}")
                return "", {
                    "status": "error",
                    "error": f"Audio conversion failed: {str(e)}",
                    "transcription_available": False
                }
            
            # Use speech recognition
            try:
                with sr.AudioFile(audio_path) as source:
                    # Adjust for ambient noise and record
                    self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.speech_recognizer.record(source)
                    
                    # Try Google's speech recognition
                    try:
                        text = self.speech_recognizer.recognize_google(audio_data, language='en-US')
                        return text, {"status": "success"}
                    except sr.UnknownValueError:
                        return "", {
                            "status": "error",
                            "error": "Speech recognition could not understand audio",
                            "transcription_available": False
                        }
                    except sr.RequestError as e:
                        return "", {
                            "status": "error",
                            "error": f"Could not request results from Google Speech Recognition service: {e}",
                            "transcription_available": False
                        }
            
            except Exception as e:
                return "", {
                    "status": "error",
                    "error": f"Error processing audio file: {str(e)}",
                    "transcription_available": False
                }
                
        except Exception as e:
            return "", {
                "status": "error",
                "error": f"Unexpected error in transcription: {str(e)}",
                "transcription_available": False
            }
            
        finally:
            # Clean up temporary file
            if tmp_wav_path and os.path.exists(tmp_wav_path):
                try:
                    os.unlink(tmp_wav_path)
                except Exception as e:
                    print(f"⚠️  Error cleaning up temporary file: {e}")

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file and return emotion classification and transcription
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            Dict containing analysis results with emotion and transcription
        """
        # Validate file
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "error": "Audio file not found",
                "audio_path": audio_path
            }
        
        try:
            # Load audio
            self._update_progress('loading_audio', 0.1, 'Loading audio file...')
            speech, sample_rate = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=speech, sr=sample_rate)
            
            # Transcribe audio
            self._update_progress('transcribing', 0.2, 'Transcribing audio...')
            transcript, transcript_info = self._transcribe_audio(audio_path)
            
            # Extract audio features
            self._update_progress('extracting_features', 0.3, 'Extracting audio features...')
            audio_features = self._extract_audio_features(speech, sample_rate)
            
            # If model not available, return basic analysis
            if self.model is None:
                self._update_progress('error', 1.0, 'Model not available')
                return {
                    "status": "error",
                    "audio_features": audio_features,
                    "duration_seconds": round(duration, 2)
                }
            
            # Prepare input
            self._update_progress('preparing_model', 0.5, 'Preparing audio for analysis...')
            inputs = self.feature_extractor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Run inference
            self._update_progress('analyzing_audio', 0.7, 'Analyzing voice patterns...')
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Calculate probabilities
            self._update_progress('processing_results', 0.85, 'Processing analysis results...')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()
            
            # Get emotion label
            emotion_name = self.emotion_labels.get(predicted_id, "unknown")
            
            # Get all emotion scores
            all_emotions = {}
            for idx, prob in enumerate(probs[0].tolist()):
                label = self.emotion_labels.get(idx, f"emotion_{idx}")
                all_emotions[label] = round(prob, 4)
            
            # Sort emotions by score
            sorted_emotions = sorted(
                all_emotions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Build response
            self._update_progress('finalizing', 0.95, 'Finalizing results...')
            response = {
                "status": "success",
                "transcription": transcript if transcript else "No transcription available",
                "transcription_status": transcript_info.get("status", "not_attempted"),
                "emotion": {
                    "dominant": emotion_name,
                    "confidence": round(confidence, 4),
                    "all_emotions": all_emotions,
                    "top_3": [
                        {
                            "emotion": emotion,
                            "confidence": round(score, 4)
                        }
                        for emotion, score in sorted_emotions[:3]
                    ]
                },
                "audio_features": {
                    "duration_seconds": round(duration, 2),
                    "energy": audio_features["energy"],
                    "zero_crossing_rate": audio_features["zero_crossing_rate"],
                    "spectral_centroid": audio_features["spectral_centroid"],
                    "tempo": audio_features["tempo"],
                    "sample_rate": sample_rate
                },
                "metadata": {
                    "model": "Wav2Vec2-XLSR",
                    "model_name": self.model_name,
                    "device": self.device,
                    "file_name": os.path.basename(audio_path),
                    "has_transcription": bool(transcript)
                }
            }
            
            return response
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "audio_path": audio_path
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
        emoji = self.emotion_emoji.get(emotion, "")
        
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


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    import json
    import sys
    
    # Initialize analyzer
    analyzer = VoiceAnalyzer()
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        # Analyze audio
        result = analyzer.analyze_audio(audio_path)
        
        # Print result
        print(json.dumps(result, indent=2))
        
        # If successful, print interpretation
        if result["status"] == "success":
            emotion = result["emotion"]["dominant"]
            confidence = result["emotion"]["confidence"]
            interpretation = analyzer.get_emotion_interpretation(emotion, confidence)
            print(f"\n{interpretation}")
    else:
        print("Usage: python voice_analyzer.py <audio_file>")
