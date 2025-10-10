from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


class TextSummarizer:
    """
    Text summarization using BART transformer model with batch processing for long texts
    """
    
    def __init__(self):
        print("Loading text summarization model...")
        # Using BART fine-tuned on CNN/DailyMail dataset
        model_name = "facebook/bart-large-cnn"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Configuration for batch processing
        self.chunk_size = 8000  # Characters per chunk
        self.overlap = 500  # Overlap between chunks to maintain context
        
        print("Text summarization model loaded successfully!")
    
    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> dict:
        """
        Summarize input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Dictionary with summary and metadata
        """
        if not text or len(text.strip()) == 0:
            return {
                "error": "Empty text provided",
                "summary": None
            }
        
        # Check if text is too short to summarize
        word_count = len(text.split())
        if word_count < 50:
            return {
                "error": "Text too short to summarize (minimum 50 words)",
                "summary": None,
                "original_length": word_count
            }
        
        # Store original length for metrics
        original_text_length = len(text)
        original_word_count = word_count
        
        try:
            # Use batch summarization for long texts
            if len(text) > 10000:
                summary = self._batch_summarize(text, max_length, min_length)
                is_batched = True
            else:
                # Single summarization for shorter texts
                result = self.pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )[0]
                summary = result['summary_text']
                is_batched = False
            
            # Calculate compression ratio
            summary_words = len(summary.split())
            compression_ratio = round((1 - summary_words / original_word_count) * 100, 1)
            
            result_dict = {
                "summary": summary,
                "original_length": original_word_count,
                "summary_length": summary_words,
                "compression_ratio": f"{compression_ratio}%",
                "model": "BART-Large-CNN (Transformer)",
                "analysis": f"Summarized {original_word_count} words into {summary_words} words ({compression_ratio}% compression)"
            }
            
            # Add batch processing info if applicable
            if is_batched:
                result_dict["batch_processed"] = True
                result_dict["original_chars"] = original_text_length
                result_dict["analysis"] += " using batch processing for long text"
            
            return result_dict
            
        except Exception as e:
            return {
                "error": f"Summarization failed: {str(e)}",
                "summary": None
            }
    
    def _split_into_chunks(self, text: str) -> list:
        """
        Split long text into overlapping chunks for batch processing
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first to maintain coherence
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _batch_summarize(self, text: str, max_length: int, min_length: int) -> str:
        """
        Summarize long text by splitting into chunks and combining summaries
        
        Args:
            text: Long input text
            max_length: Maximum length per summary
            min_length: Minimum length per summary
            
        Returns:
            Combined summary string
        """
        # Split text into chunks
        chunks = self._split_into_chunks(text)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk.split()) < 50:
                continue
                
            try:
                result = self.pipeline(
                    chunk,
                    max_length=max_length,
                    min_length=min_length // 2,  # Shorter min for chunks
                    do_sample=False,
                    truncation=True
                )[0]
                chunk_summaries.append(result['summary_text'])
            except Exception as e:
                print(f"Warning: Failed to summarize chunk {i+1}: {str(e)}")
                continue
        
        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # If combined summary is still very long, summarize it again
        if len(combined_summary.split()) > max_length * 2:
            try:
                final_result = self.pipeline(
                    combined_summary,
                    max_length=max_length * 2,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )[0]
                return final_result['summary_text']
            except Exception:
                # If final summarization fails, return combined summary
                return combined_summary
        
        return combined_summary
