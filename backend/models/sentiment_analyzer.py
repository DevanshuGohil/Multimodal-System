"""
Sentiment Analysis using Groq API
==================================

Fast sentiment analysis using Groq's Llama models for text classification.
Provides 3-class sentiment detection: Positive, Negative, and Neutral.

Dependencies:
    pip install requests python-dotenv
"""

import os
import requests
from typing import Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class SentimentAnalyzer:
    """
    Sentiment Analyzer using Groq API
    
    Uses Groq's Llama models for fast and accurate sentiment classification.
    Returns sentiment with confidence scores and emotional interpretation.
    
    Usage:
        analyzer = SentimentAnalyzer(api_key="your_groq_api_key")
        result = analyzer.analyze("I love this product!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize Sentiment Analyzer
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY in environment)
            model: Groq model to use for analysis
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required. Set it in environment or pass to constructor.")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Emotion mappings
        self.emotion_mapping = {
            "positive": {
                "high": "Very Happy 😊",
                "medium": "Happy 🙂",
                "low": "Slightly Positive 😌"
            },
            "negative": {
                "high": "Very Sad 😢",
                "medium": "Sad 😔",
                "low": "Slightly Negative 😕"
            },
            "neutral": {
                "high": "Neutral 😐",
                "medium": "Slightly Neutral 😶",
                "low": "Mixed Feelings 😑"
            }
        }
        
        print(f"✅ Sentiment Analyzer initialized with Groq API ({self.model})")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of input text using Groq API
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment analysis results:
            {
                "status": "success",
                "sentiment": "Positive",
                "confidence": 95.0,
                "emotion": "Very Happy 😊",
                "analysis": "...",
                "metadata": {...}
            }
        """
        
        # Validate input
        if not text or len(text.strip()) == 0:
            return {
                "status": "error",
                "error": "Empty text provided",
                "sentiment": "Neutral",
                "confidence": 0.0
            }
        
        # Truncate if too long
        original_length = len(text)
        if len(text) > 5000:
            text = text[:5000]
        
        try:
            # Call Groq API
            result = self._call_groq_api(text)
            
            if result["status"] == "error":
                return result
            
            # Parse sentiment and confidence
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            # Get emotion label
            emotion = self._get_emotion_label(sentiment, confidence)
            
            # Get detailed analysis
            analysis = self._get_detailed_analysis(sentiment, confidence)
            
            return {
                "status": "success",
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "emotion": emotion,
                "text_length": original_length,
                "analysis": analysis,
                "metadata": {
                    "model": self.model,
                    "provider": "Groq API",
                    "truncated": original_length > 5000
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "sentiment": "Neutral",
                "confidence": 0.0
            }
    
    def _call_groq_api(self, text: str) -> Dict[str, Any]:
        """
        Call Groq API for sentiment classification
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment and confidence
        """
        
        # Craft prompt for sentiment analysis
        system_prompt = """You are a sentiment analysis expert. Analyze the sentiment of the given text and respond in EXACTLY this JSON format:

{
  "sentiment": "Positive" or "Negative" or "Neutral",
  "confidence": 0-100 (number only),
  "reasoning": "brief explanation"
}

Rules:
- sentiment must be EXACTLY one of: "Positive", "Negative", or "Neutral" (capitalize first letter)
- confidence must be a number between 0 and 100
- Be precise and consistent
- Consider context, sarcasm, and nuance"""

        user_prompt = f"Analyze the sentiment of this text:\n\n{text}"
        
        try:
            response = requests.post(
                self.base_url,
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
                    "temperature": 0.1,  # Low temperature for consistent results
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"}  # Force JSON output
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            import json
            parsed = json.loads(content)
            
            # Validate and normalize
            sentiment = parsed.get("sentiment", "Neutral")
            confidence = float(parsed.get("confidence", 50))
            
            # Ensure sentiment is properly capitalized
            sentiment_map = {
                "positive": "Positive",
                "negative": "Negative",
                "neutral": "Neutral"
            }
            sentiment = sentiment_map.get(sentiment.lower(), sentiment)
            
            # Clamp confidence to 0-100
            confidence = max(0, min(100, confidence))
            
            return {
                "status": "success",
                "sentiment": sentiment,
                "confidence": confidence,
                "reasoning": parsed.get("reasoning", "")
            }
            
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": "Groq API timeout"
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"Groq API error: {str(e)}"
            }
        
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return {
                "status": "error",
                "error": f"Failed to parse response: {str(e)}"
            }
    
    def _get_emotion_label(self, sentiment: str, confidence: float) -> str:
        """
        Get emotion label based on sentiment and confidence
        
        Args:
            sentiment: Sentiment classification
            confidence: Confidence score (0-100)
            
        Returns:
            Emotion label with emoji
        """
        sentiment_key = sentiment.lower()
        
        if confidence >= 85:
            level = "high"
        elif confidence >= 65:
            level = "medium"
        else:
            level = "low"
        
        return self.emotion_mapping.get(sentiment_key, {}).get(level, "Neutral 😐")
    
    def _get_detailed_analysis(self, sentiment: str, confidence: float) -> str:
        """
        Generate detailed analysis text
        
        Args:
            sentiment: Sentiment classification
            confidence: Confidence score
            
        Returns:
            Analysis description
        """
        if confidence >= 85:
            certainty = "very confident"
        elif confidence >= 70:
            certainty = "confident"
        elif confidence >= 55:
            certainty = "moderately confident"
        else:
            certainty = "somewhat uncertain"
        
        return f"The model is {certainty} that this text expresses a {sentiment.lower()} sentiment."


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Factory function to get SentimentAnalyzer instance"""
    return SentimentAnalyzer()