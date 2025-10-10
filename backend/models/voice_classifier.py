import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import numpy as np


class VoiceClassifier:
    """
    Voice emotion classification using Wav2Vec2 transformer model
    """
    
    def __init__(self):
        print("Loading voice classification model...")
        # Using Wav2Vec2 fine-tuned for emotion recognition
        # This is a transformer-based model for audio
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        
        try:
            # Use FeatureExtractor instead of Processor (no tokenizer needed for audio)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Emotion labels for this model
            self.emotion_labels = {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "neutral",
                5: "sad",
                6: "surprise"
            }
            
            # Emotion emojis
            self.emotion_emoji = {
                "angry": "😠",
                "disgust": "🤢",
                "fear": "😨",
                "happy": "😊",
                "neutral": "😐",
                "sad": "😢",
                "surprise": "😮"
            }
            
            print("Voice classification model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to basic audio analysis...")
            self.model = None
    
    def classify(self, audio_path: str) -> dict:
        """
        Classify emotion from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion classification results
        """
        try:
            # Load audio file
            speech, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Get audio features for metadata
            duration = librosa.get_duration(y=speech, sr=sample_rate)
            
            # Basic audio analysis
            audio_features = self._extract_audio_features(speech, sample_rate)
            
            if self.model is None:
                return {
                    "error": "Model not available",
                    "audio_features": audio_features,
                    "duration": round(duration, 2)
                }
            
            # Prepare input for model using feature extractor
            inputs = self.feature_extractor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()
            
            # Get emotion label
            emotion = self.emotion_labels.get(predicted_id, "unknown")
            emoji = self.emotion_emoji.get(emotion, "")
            
            # Get all emotion scores
            all_emotions = {}
            for idx, prob in enumerate(probs[0].tolist()):
                emotion_name = self.emotion_labels.get(idx, f"emotion_{idx}")
                all_emotions[emotion_name] = round(prob * 100, 2)
            
            return {
                "emotion": f"{emotion.capitalize()} {emoji}",
                "confidence": round(confidence * 100, 2),
                "all_emotions": all_emotions,
                "audio_features": audio_features,
                "duration": round(duration, 2),
                "model": "Wav2Vec2-XLSR (Transformer)",
                "analysis": f"Detected {emotion} emotion with {round(confidence * 100, 1)}% confidence"
            }
            
        except Exception as e:
            return {
                "error": f"Voice analysis failed: {str(e)}",
                "suggestion": "Please upload a valid audio file (WAV, MP3, etc.)"
            }
    
    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Extract basic audio features"""
        try:
            # Calculate features
            rms = librosa.feature.rms(y=audio)[0]
            energy = float(np.mean(rms))
            
            # Zero crossing rate (indicator of voice characteristics)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral centroid (brightness of sound)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            
            return {
                "energy": round(energy, 4),
                "zero_crossing_rate": round(zcr_mean, 4),
                "spectral_centroid": round(spectral_centroid_mean, 2),
                "tempo": round(float(tempo), 2),
                "sample_rate": sample_rate
            }
        except Exception as e:
            return {
                "error": f"Feature extraction failed: {str(e)}"
            }
