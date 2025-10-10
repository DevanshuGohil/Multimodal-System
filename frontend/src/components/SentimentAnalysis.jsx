import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const SentimentAnalysis = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('text', text);

      const response = await axios.post('http://localhost:8000/api/sentiment', formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment?.includes('Positive')) return 'from-green-500 to-emerald-500';
    if (sentiment?.includes('Negative')) return 'from-red-500 to-pink-500';
    return 'from-gray-500 to-slate-500';
  };

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20"
      >
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
          Sentiment Analysis with RoBERTa
        </h2>
        <p className="text-gray-300 text-sm mb-6">
          Using Twitter-RoBERTa model with native 3-class sentiment detection (Positive, Negative, Neutral)
        </p>
        
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2 text-gray-300">
            Enter text to analyze
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste your text here... (e.g., 'I absolutely love this product! It's amazing!')"
            className="w-full h-40 px-4 py-3 bg-white/5 border border-white/20 rounded-xl 
                     text-white placeholder-gray-500 focus:outline-none focus:ring-2 
                     focus:ring-blue-500 focus:border-transparent resize-none"
          />
          <p className="text-sm text-gray-400 mt-2">
            {text.length} characters
          </p>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-3 px-6 
                   rounded-xl font-semibold flex items-center justify-center gap-2
                   hover:shadow-lg hover:shadow-blue-500/50 transition-all duration-300
                   disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              Analyze Sentiment
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

        {result && !result.error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 space-y-6"
          >
            <div className={`bg-gradient-to-r ${getSentimentColor(result.sentiment)} p-6 rounded-xl`}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">
                    {result.emotion}
                  </h3>
                  <p className="text-white/80">{result.sentiment} Sentiment</p>
                </div>
                <CheckCircle className="w-12 h-12 text-white" />
              </div>
              <div className="bg-white/20 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-white font-medium">Confidence</span>
                  <span className="text-white font-bold">{result.confidence}%</span>
                </div>
                <div className="w-full bg-white/30 rounded-full h-3">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className="bg-white h-3 rounded-full"
                  />
                </div>
              </div>
            </div>

            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h4 className="text-lg font-semibold mb-3 text-gray-200">Analysis Details</h4>
              <div className="space-y-2 text-gray-300">
                <p><span className="font-medium">Model:</span> {result.model}</p>
                <p><span className="font-medium">Text Length:</span> {result.text_length} characters</p>
                <p className="text-sm text-gray-400 mt-4">{result.analysis}</p>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default SentimentAnalysis;
