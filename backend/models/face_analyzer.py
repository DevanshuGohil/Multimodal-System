from deepface import DeepFace
import cv2
import numpy as np
from typing import Dict, List


class FaceAnalyzer:
    """
    Face analysis using DeepFace (includes multiple transformer-based models)
    Detects: Age, Gender, Emotion, Race
    """
    
    def __init__(self):
        print("Initializing face analysis model...")
        # DeepFace uses multiple models including VGG-Face, Facenet, etc.
        # It will download models on first use
        self.actions = ['age', 'gender', 'emotion', 'race']
        print("Face analysis model initialized!")
    
    def analyze(self, image_path: str) -> dict:
        """
        Analyze face in image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    "error": "Could not read image file",
                    "faces_detected": 0
                }
            
            # Analyze face using DeepFace
            # This uses transformer-based models for emotion detection
            result = DeepFace.analyze(
                img_path=image_path,
                actions=self.actions,
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                faces = result
            else:
                faces = [result]
            
            # Process results
            analyzed_faces = []
            for idx, face in enumerate(faces):
                face_data = {
                    "face_number": idx + 1,
                    "age": int(face.get('age', 0)),
                    "gender": self._format_gender(face.get('gender', {})),
                    "emotion": self._format_emotion(face.get('emotion', {})),
                    "race": self._format_race(face.get('race', {})),
                    "region": face.get('region', {})
                }
                analyzed_faces.append(face_data)
            
            return {
                "faces_detected": len(analyzed_faces),
                "faces": analyzed_faces,
                "model": "DeepFace (CNN + Transformer-based)",
                "analysis": f"Detected {len(analyzed_faces)} face(s) in the image"
            }
            
        except ValueError as e:
            # No face detected
            return {
                "error": "No face detected in the image",
                "faces_detected": 0,
                "suggestion": "Please upload an image with a clear, visible face"
            }
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "faces_detected": 0
            }
    
    def _format_gender(self, gender_dict: dict) -> dict:
        """Format gender prediction with confidence"""
        if not gender_dict:
            return {"prediction": "Unknown", "confidence": 0}
        
        # Get dominant gender
        dominant = max(gender_dict.items(), key=lambda x: x[1])
        return {
            "prediction": dominant[0],
            "confidence": round(dominant[1], 2),
            "all_scores": {k: round(v, 2) for k, v in gender_dict.items()}
        }
    
    def _format_emotion(self, emotion_dict: dict) -> dict:
        """Format emotion prediction with confidence"""
        if not emotion_dict:
            return {"prediction": "Unknown", "confidence": 0}
        
        # Get dominant emotion
        dominant = max(emotion_dict.items(), key=lambda x: x[1])
        
        # Map emotions to emojis
        emotion_emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "surprise": "😮",
            "fear": "😨",
            "disgust": "🤢",
            "neutral": "😐"
        }
        
        emotion_name = dominant[0]
        emoji = emotion_emoji.get(emotion_name, "")
        
        return {
            "prediction": f"{emotion_name.capitalize()} {emoji}",
            "confidence": round(dominant[1], 2),
            "all_scores": {k: round(v, 2) for k, v in emotion_dict.items()}
        }
    
    def _format_race(self, race_dict: dict) -> dict:
        """Format race prediction with confidence"""
        if not race_dict:
            return {"prediction": "Unknown", "confidence": 0}
        
        # Get dominant race
        dominant = max(race_dict.items(), key=lambda x: x[1])
        return {
            "prediction": dominant[0],
            "confidence": round(dominant[1], 2),
            "all_scores": {k: round(v, 2) for k, v in race_dict.items()}
        }
