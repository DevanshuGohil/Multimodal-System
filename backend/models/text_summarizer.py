import os
import re
from typing import Dict, Optional
import groq
from groq import Groq



class TextSummarizer:
    """
    Text summarization using Groq API for fast and efficient summarization
    """
    
    def __init__(self, api_key: Optional[str] = None):
        print("Initializing Groq text summarizer...")
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment variable or pass it to the constructor."
            )
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Model configuration - Using the recommended model
        self.model = "llama-3.3-70b-versatile"  # Groq's Mixtral model with 32k context length (FP16 precision)
        
        # Configuration for chunking long texts
        self.chunk_size = 30000  # Characters per chunk (well under the 32k token limit)
        self.overlap = 1000  # Overlap between chunks to maintain context
        
        print("Groq text summarizer initialized successfully!")
    
    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> dict:
        """
        Summarize input text using Groq API
        
        Args:
            text: Input text to summarize
            max_length: Approximate maximum length of summary (in words, default: 300)
            min_length: Approximate minimum length of summary (in words, default: 100)
            
        Returns:
            Dictionary with summary and metadata
        """
        if not text or len(text.strip()) == 0:
            return {
                "error": "Empty text provided",
                "summary": ""
            }
        
        # Clean and prepare the text
        text = text.strip()
        original_word_count = len(text.split())
        
        # If text is very short, return as is
        if original_word_count < 30:
            return {
                "summary": text,
                "original_length": original_word_count,
                "summary_length": original_word_count,
                "compression_ratio": "0%",
                "model": self.model,
                "analysis": "Text too short for meaningful summarization"
            }
        
        try:
            # For very long texts, use chunking
            if len(text) > self.chunk_size:
                summary = self._batch_summarize(text, max_length, min_length)
            else:
                # For shorter texts, summarize in one go
                prompt = self._create_summary_prompt(text, max_length, min_length)
                summary = self._call_groq_api(prompt)
            
            # Clean up the summary
            summary = self._clean_summary(summary)
            summary_word_count = len(summary.split())
            
            # Calculate compression ratio
            compression_ratio = round((1 - (summary_word_count / original_word_count)) * 100, 2) if original_word_count > 0 else 0
            
            return {
                "summary": summary,
                "original_length": original_word_count,
                "summary_length": summary_word_count,
                "compression_ratio": f"{compression_ratio}%",
                "model": self.model,
                "analysis": f"Summarized {original_word_count} words into {summary_word_count} words ({compression_ratio}% compression)"
            }
            
        except Exception as e:
            return {
                "error": f"Error generating summary: {str(e)}",
                "summary": ""
            }
    
    def _create_summary_prompt(self, text: str, max_length: int, min_length: int) -> str:
        """
        Create a prompt for the Groq API to generate a summary
        
        Args:
            text: Text to summarize
            max_length: Approximate max length of summary in words
            min_length: Approximate min length of summary in words
            
        Returns:
            Formatted prompt string
        """
        return f"""Please provide a concise and coherent summary of the following text. 
The summary should be between {min_length} and {max_length} words, capturing the main points and key details.

Text to summarize:
{text}

Summary:"""

    def _call_groq_api(self, prompt: str) -> str:
        """
        Call the Groq API to generate a summary
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Generated summary text
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that creates concise and accurate summaries of text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more deterministic outputs
                max_tokens=4000,  # Max tokens for the response
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg.lower():
                raise Exception("API rate limit exceeded. Please try again later.")
            elif "authentication" in error_msg.lower():
                raise Exception("Authentication failed. Please check your API key.")
            else:
                raise Exception(f"API error: {error_msg}")

    def _clean_summary(self, summary: str) -> str:
        """
        Clean up the generated summary
        
        Args:
            summary: Raw summary text
            
        Returns:
            Cleaned summary text
        """
        # Remove any leading/trailing whitespace
        summary = summary.strip()
        
        # Remove any quotation marks that might be around the summary
        summary = summary.strip('"\'')
        
        # Ensure the summary ends with proper punctuation
        if summary and summary[-1] not in {'.', '!', '?'}:
            summary += '.'
            
        return summary

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
            max_length: Approximate maximum length per summary in words
            min_length: Approximate minimum length per summary in words
            
        Returns:
            Combined summary string
        """
        # Split text into chunks
        chunks = self._split_into_chunks(text)
        
        # Adjust summary length for chunks (shorter than final target)
        chunk_max_length = max(min_length, max_length // 2)
        chunk_min_length = max(min_length // 2, 30)  # Ensure minimum length
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk.split()) < 30:  # Lower threshold for Groq
                continue
                
            try:
                prompt = self._create_summary_prompt(chunk, chunk_max_length, chunk_min_length)
                summary = self._call_groq_api(prompt)
                chunk_summaries.append(self._clean_summary(summary))
                
                # Print progress
                print(f"Processed chunk {i+1}/{len(chunks)}")
                
            except Exception as e:
                print(f"Warning: Failed to summarize chunk {i+1}: {str(e)}")
                # Include the chunk text as fallback if summarization fails
                chunk_summaries.append(chunk[:500] + "...")
                continue
        
        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        
        # If combined summary is still very long, summarize it again
        if len(combined_summary.split()) > max_length * 1.5:
            try:
                final_prompt = self._create_summary_prompt(
                    combined_summary, 
                    max_length, 
                    min_length
                )
                final_summary = self._call_groq_api(final_prompt)
                return self._clean_summary(final_summary)
            except Exception as e:
                print(f"Warning: Failed to summarize combined chunks: {str(e)}")
                # If final summarization fails, return combined summary
                return combined_summary
        
        return combined_summary
