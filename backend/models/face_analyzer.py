import os
import base64
import json
import requests
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageFile
import io
from dotenv import load_dotenv

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load environment variables
load_dotenv()


class FaceAnalyzer:
    """
    Face analysis using Face++ API
    Detects: Age, Gender, Emotion, and other facial attributes
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        print("Initializing face analysis with Face++ API...")
        self.api_key = api_key or os.getenv("FACEPP_API_KEY")
        self.api_secret = api_secret or os.getenv("FACEPP_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Face++ API key and secret not provided. "
                "Set FACEPP_API_KEY and FACEPP_API_SECRET environment variables or pass them to the constructor."
            )
        self.base_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        print("Face analysis initialized with Face++ API!")
    
    def analyze(self, image_path: str) -> dict:
        """
        Analyze faces in image using Face++ API
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results for all detected faces
        """
        try:
            # Read and validate image
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
            
            # Convert to base64 for API
            img_str = base64.b64encode(img_data).decode('utf-8')
            
            # Get all faces with attributes
            faces_data = self._get_all_faces_attributes(image_path)
            
            if not faces_data:
                # If no faces detected, try with the full image
                print("No faces detected in the initial analysis, trying with the full image...")
                img = cv2.imread(image_path)
                if img is not None:
                    # Convert to RGB and encode
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, buffer = cv2.imencode('.jpg', img_rgb)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    faces_data = self._get_all_faces_attributes(image_path, img_str)
            
            if faces_data:
                return {
                    "faces_detected": len(faces_data),
                    "faces": faces_data,
                    "model": "Face++ API",
                    "analysis": f"Successfully analyzed {len(faces_data)} face(s)"
                }
            else:
                return {
                    "error": "No faces detected in the image",
                    "faces_detected": 0,
                    "suggestion": "Please upload an image with clear, visible faces"
                }
                
        except Exception as e:
            print(f"Error in face analysis: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "faces_detected": 0
            }
    
    def _process_image_for_api(self, image_path: str) -> Optional[io.BytesIO]:
        """Process image to ensure it's in a format accepted by Face++ API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BytesIO object with processed image or None if processing fails
        """
        try:
            # Read the image file
            with open(image_path, 'rb') as f:
                img_data = f.read()
            
            # Try to open and verify the image
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if the image is too large (Face++ has a size limit)
            max_size = 4096
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save to a BytesIO object in JPEG format
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95)
            output.seek(0)
            
            return output
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def _get_all_faces_attributes(self, image_path: str, image_base64: str = None) -> List[dict]:
        """Get attributes for all faces in the image using Face++ API
        
        Args:
            image_path: Path to the image file
            image_base64: Optional base64 encoded image string if already processed
            
        Returns:
            List of dictionaries containing face attributes for each face
        """
        try:
            # Process the image for the API
            processed_img = self._process_image_for_api(image_path)
            if not processed_img:
                print("Failed to process image for Face++ API")
                return []
            
            # Prepare the file for upload
            files = {
                'image_file': ('processed.jpg', processed_img, 'image/jpeg')
            }
            
            # Prepare the request data
            data = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'return_attributes': 'gender,age,emotion,beauty,facequality',
                'return_landmark': 1
            }
            
            print(f"Sending request to Face++ API for {os.path.basename(image_path)}")
            
            # Make the API request
            response = requests.post(
                self.base_url,
                files=files,
                data=data,
                timeout=30
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            result = response.json()
            
            # Debug: Print the API response
            print("Face++ API Response:")
            print(json.dumps(result, indent=2))
            
            faces_data = []
            
            # Process the response
            if 'faces' in result and result['faces']:
                print(f"Found {len(result['faces'])} faces in the image")
                
                for i, face in enumerate(result['faces'], 1):
                    try:
                        # Get face rectangle
                        rect = face.get('face_rectangle', {})
                        
                        # Get attributes
                        attrs = face.get('attributes', {})
                        
                        # Get gender
                        gender = attrs.get('gender', {})
                        gender_val = gender.get('value', 'Unknown').capitalize()
                        
                        # Get age
                        age = attrs.get('age', {})
                        age_val = str(age.get('value', 'Unknown'))
                        
                        # Get emotion
                        emotion_data = attrs.get('emotion', {})
                        if emotion_data:
                            # Get the emotion with highest confidence
                            emotion_type = max(emotion_data.items(), 
                                            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                            emotion = {
                                'prediction': emotion_type[0].capitalize(),
                                'confidence': float(emotion_type[1]) / 100.0,
                                'all_scores': {k.capitalize(): float(v) for k, v in emotion_data.items()}
                            }
                        else:
                            emotion = self._get_default_emotion()
                        
                        # Create face data
                        face_data = {
                            'face_number': i,
                            'age': age_val,
                            'gender': gender_val,
                            'emotion': emotion,
                            'region': {
                                'x': rect.get('left', 0),
                                'y': rect.get('top', 0),
                                'w': rect.get('width', 0),
                                'h': rect.get('height', 0)
                            },
                            'confidence': float(face.get('face_probability', 0))
                        }
                        
                        faces_data.append(face_data)
                        
                    except Exception as face_error:
                        print(f"Error processing face {i}: {str(face_error)}")
                        continue
            
            return faces_data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Face++ API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus Code: {e.response.status_code}"
                try:
                    error_msg += f"\nResponse: {e.response.text}"
                except:
                    pass
            print(error_msg)
            return []
            
        except Exception as e:
            print(f"Unexpected error in face analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_face_attributes(self, image_base64: str) -> dict:
        """Get face attributes for a single face (for backward compatibility)
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Dictionary containing age, gender, emotion, and confidence
        """
        # Save base64 to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(base64.b64decode(image_base64))
            temp_path = temp_file.name
        
        try:
            faces = self._get_all_faces_attributes(temp_path)
            if faces:
                return {
                    'age': faces[0].get('age', 'Unknown'),
                    'gender': faces[0].get('gender', 'Unknown'),
                    'emotion': faces[0].get('emotion', {}).get('prediction', 'Unknown'),
                    'confidence': faces[0].get('confidence', 0)
                }
            return {
                'age': 'Unknown',
                'gender': 'Unknown',
                'emotion': 'Unknown',
                'confidence': 0
            }
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        try:
            # Initialize default response
            attributes = {
                "age": "Unknown",
                "gender": "Unknown",
                "emotion": "Unknown",
                "confidence": 0
            }
            
            # Convert base64 to image file
            try:
                image_data = base64.b64decode(image_base64)
                image_file = io.BytesIO(image_data)
                image_file.name = 'face.jpg'  # Required for requests
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return attributes
            
            # Prepare request to Face++ API
            try:
                files = {'image_file': image_file}
                data = {
                    'api_key': self.api_key,
                    'api_secret': self.api_secret,
                    'return_attributes': 'gender,age,emotion,beauty'
                }
                
                response = requests.post(
                    self.base_url,
                    files=files,
                    data=data,
                    timeout=30
                )
                
                response.raise_for_status()
                result = response.json()
                
                if 'faces' in result and len(result['faces']) > 0:
                    face = result['faces'][0]
                    attributes['confidence'] = face.get('face_rectangle', {}).get('confidence', 0)
                    
                    # Extract attributes
                    attrs = face.get('attributes', {})
                    
                    # Get gender
                    gender = attrs.get('gender', {})
                    if gender.get('value'):
                        attributes['gender'] = gender['value'].capitalize()
                    
                    # Get age
                    age = attrs.get('age', {})
                    if age.get('value'):
                        attributes['age'] = str(age['value'])
                    
                    # Get emotion
                    emotion = attrs.get('emotion', {})
                    if emotion:
                        # Get the emotion with highest confidence
                        emotion_type = max(emotion.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                        if emotion_type[1] > 0:
                            attributes['emotion'] = emotion_type[0].capitalize()
                            attributes['confidence'] = max(attributes['confidence'], emotion_type[1] / 100)
                
                return attributes
                
            except requests.exceptions.RequestException as e:
                print(f"Error calling Face++ API: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return attributes
                
        except Exception as e:
            print(f"Unexpected error in face analysis: {str(e)}")
            return attributes
    
    def _get_emotion_emoji(self) -> dict:
        """Returns a dictionary mapping emotions to emojis"""
        return {
            'happy': '😊', 'sad': '😢', 'angry': '😠', 'fear': '😨',
            'surprise': '😲', 'neutral': '😐', 'disgust': '🤢',
            'contempt': '😏', 'disappointment': '😞', 'confusion': '😕',
            'awe': '😮', 'excitement': '🤩', 'amusement': '😄',
            'contentment': '😌', 'embarrassment': '😳', 'guilt': '😔',
            'pride': '😌', 'relief': '😌', 'shame': '😔', 
            'suffering': '😣', 'triumph': '😤'
        }

    def _detect_emotion(self, image_base64: str) -> dict:
        """Detect emotion in face using Face++ API
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Dictionary containing emotion prediction, confidence, and all scores
        """
        try:
            # Save base64 to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(base64.b64decode(image_base64))
                temp_path = temp_file.name
            
            faces = self._get_all_faces_attributes(temp_path)
            if faces and 'emotion' in faces[0]:
                emotion_result = faces[0]['emotion']
                # Add emoji to the emotion prediction
                emoji = self._get_emotion_emoji().get(emotion_result.get('prediction', '').lower(), '')
                if emoji:
                    emotion_result['prediction'] = f"{emotion_result['prediction']} {emoji}"
                return emotion_result
                
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            
        return self._get_default_emotion()
    
    def _get_default_emotion_response(self) -> dict:
        """Return a default emotion response when detection fails"""
        print("Using default emotion response")
        return {
            "prediction": "Unknown",
            "confidence": 0,
            "all_scores": {}
            }
