import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Loader2, AlertCircle, CheckCircle, Copy } from 'lucide-react';
import axios from 'axios';
import { API_ENDPOINTS } from '../config/api';

const TextSummarization = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const handleSummarize = async () => {
    if (!text.trim()) {
      setError('Please enter some text to summarize');
      return;
    }

    const wordCount = text.split(/\s+/).length;
    if (wordCount < 50) {
      setError('Text is too short to summarize. Please enter at least 50 words.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('max_length', '130');
      formData.append('min_length', '30');

      const response = await axios.post(API_ENDPOINTS.summarize, formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to summarize text');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (result?.summary) {
      navigator.clipboard.writeText(result.summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const sampleText = `Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.`;

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20"
      >
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
          Text Summarization with BART
        </h2>
        
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <label className="block text-sm font-medium text-gray-300">
              Enter text to summarize (minimum 50 words)
            </label>
            <button
              onClick={() => setText(sampleText)}
              className="text-xs text-purple-400 hover:text-purple-300 underline"
            >
              Use sample text
            </button>
          </div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a long article, essay, or document here..."
            className="w-full h-48 px-4 py-3 bg-white/5 border border-white/20 rounded-xl 
                     text-white placeholder-gray-500 focus:outline-none focus:ring-2 
                     focus:ring-purple-500 focus:border-transparent resize-none"
          />
          <p className="text-sm text-gray-400 mt-2">
            {text.split(/\s+/).filter(w => w.length > 0).length} words • {text.length} characters
          </p>
        </div>

        <button
          onClick={handleSummarize}
          disabled={loading || !text.trim()}
          className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 
                   rounded-xl font-semibold flex items-center justify-center gap-2
                   hover:shadow-lg hover:shadow-purple-500/50 transition-all duration-300
                   disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Summarizing...
            </>
          ) : (
            <>
              <FileText className="w-5 h-5" />
              Summarize Text
            </>
          )}
        </button>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl flex items-start gap-3"
          >
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-red-300">{error}</p>
          </motion.div>
        )}

        {result && !result.error && result.summary && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 space-y-6"
          >
            <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-6 rounded-xl">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-2xl font-bold text-white">Summary</h3>
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 
                           rounded-lg transition-colors duration-200"
                >
                  {copied ? (
                    <>
                      <CheckCircle className="w-4 h-4 text-white" />
                      <span className="text-white text-sm">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4 text-white" />
                      <span className="text-white text-sm">Copy</span>
                    </>
                  )}
                </button>
              </div>
              <div className="bg-white/20 rounded-lg p-4">
                <p className="text-white leading-relaxed">{result.summary}</p>
              </div>
            </div>

            {/* Batch Processing Badge */}
            {result.batch_processed && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-blue-500/20 border border-blue-500/50 rounded-xl p-4"
              >
                <div className="flex items-center gap-3">
                  <div className="bg-blue-500 rounded-full p-2">
                    <CheckCircle className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="text-blue-300 font-semibold">Batch Processing Used</p>
                    <p className="text-blue-200/80 text-sm">
                      Long text ({result.original_chars?.toLocaleString()} characters) was split into chunks for optimal summarization
                    </p>
                  </div>
                </div>
              </motion.div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <p className="text-gray-400 text-sm mb-1">Original Length</p>
                <p className="text-2xl font-bold text-white">{result.original_length} words</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <p className="text-gray-400 text-sm mb-1">Summary Length</p>
                <p className="text-2xl font-bold text-white">{result.summary_length} words</p>
              </div>
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <p className="text-gray-400 text-sm mb-1">Compression</p>
                <p className="text-2xl font-bold text-green-400">{result.compression_ratio}</p>
              </div>
            </div>

            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h4 className="text-lg font-semibold mb-3 text-gray-200">Model Information</h4>
              <div className="space-y-2 text-gray-300">
                <p><span className="font-medium">Model:</span> {result.model}</p>
                {result.batch_processed && (
                  <p><span className="font-medium">Processing:</span> Batch Summarization (Chunked)</p>
                )}
                <p className="text-sm text-gray-400 mt-4">{result.analysis}</p>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default TextSummarization;
