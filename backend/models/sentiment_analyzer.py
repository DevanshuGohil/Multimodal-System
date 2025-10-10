from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class SentimentAnalyzer:
    """
    Sentiment analysis using RoBERTa transformer model
    
    Uses twitter-roberta-base-sentiment-latest which natively supports
    3-class sentiment classification: Positive, Negative, and Neutral.
    """
    
    def __init__(self):
        print("Loading sentiment analysis model...")
        # Using RoBERTa fine-tuned on Twitter data with 3 classes (positive, negative, neutral)
        # This model natively supports neutral sentiment detection
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Label mapping for this model
        # The twitter-roberta model uses these labels
        self.label_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral", 
            "LABEL_2": "Positive",
            # Also handle lowercase versions
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
            # And direct names if model outputs them
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
            "NEGATIVE": "Negative",
            "NEUTRAL": "Neutral",
            "POSITIVE": "Positive"
        }
        
        print("Sentiment analysis model loaded successfully!")
    
    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        if not text or len(text.strip()) == 0:
            return {
                "error": "Empty text provided",
                "sentiment": None,
                "confidence": 0.0
            }
        
        # Truncate text if too long (DistilBERT max length is 512 tokens)
        if len(text) > 5000:
            text = text[:5000]
        
        # Get prediction
        result = self.pipeline(text)[0]
        
        # Extract raw prediction
        raw_label = result['label']
        confidence = round(result['score'] * 100, 2)
        
        # Map model labels to sentiment names
        sentiment = self.label_mapping.get(raw_label, raw_label)
        
        # Determine emotion based on sentiment and confidence
        if sentiment == "Neutral":
            if confidence > 70:
                emotion = "Neutral 😐"
            else:
                emotion = "Slightly Neutral 😶"
        elif sentiment == "Positive":
            if confidence > 90:
                emotion = "Very Happy 😊"
            elif confidence > 70:
                emotion = "Happy 🙂"
            else:
                emotion = "Slightly Positive 😌"
        elif sentiment == "Negative":
            if confidence > 90:
                emotion = "Very Sad 😢"
            elif confidence > 70:
                emotion = "Sad 😔"
            else:
                emotion = "Slightly Negative 😕"
        else:
            emotion = "Neutral 😐"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "emotion": emotion,
            "text_length": len(text),
            "model": "RoBERTa-Base (Transformer)",
            "analysis": self._get_detailed_analysis(sentiment, confidence)
        }
    
    def _get_detailed_analysis(self, sentiment: str, confidence: float) -> str:
        """Generate detailed analysis text"""
        if confidence > 85:
            certainty = "very confident"
        elif confidence > 70:
            certainty = "confident"
        elif confidence > 55:
            certainty = "moderately confident"
        else:
            certainty = "somewhat uncertain"
        
        return f"The model is {certainty} that this text expresses a {sentiment.lower()} sentiment."
