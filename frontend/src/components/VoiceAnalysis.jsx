import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, Loader2, AlertCircle, Mic, Music } from 'lucide-react';
import axios from 'axios';
import { API_ENDPOINTS } from '../config/api';

const VoiceAnalysis = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith('audio/')) {
        setError('Please select a valid audio file');
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select an audio file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(API_ENDPOINTS.voiceAnalysis, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Failed to analyze voice');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      setFile(droppedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const getEmotionColor = (emotion) => {
    if (emotion?.toLowerCase().includes('happy')) return 'from-green-500 to-emerald-500';
    if (emotion?.toLowerCase().includes('sad')) return 'from-blue-500 to-cyan-500';
    if (emotion?.toLowerCase().includes('angry')) return 'from-red-500 to-orange-500';
    if (emotion?.toLowerCase().includes('fear')) return 'from-purple-500 to-pink-500';
    if (emotion?.toLowerCase().includes('surprise')) return 'from-yellow-500 to-orange-500';
    return 'from-gray-500 to-slate-500';
  };

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20"
      >
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
          Voice Emotion Analysis with Wav2Vec2
        </h2>
        
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          className="mb-6 border-2 border-dashed border-white/30 rounded-xl p-8 text-center 
                   cursor-pointer hover:border-green-400 hover:bg-white/5 transition-all duration-300"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="hidden"
          />
          
          {file ? (
            <div className="space-y-4">
              <Music className="w-16 h-16 mx-auto text-green-400" />
              <div>
                <p className="text-lg text-gray-300 mb-2">{file.name}</p>
                <p className="text-sm text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <p className="text-sm text-gray-400 mt-2">Click to change file</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="w-16 h-16 mx-auto text-gray-400" />
              <div>
                <p className="text-lg text-gray-300 mb-2">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-gray-500">
                  WAV, MP3, M4A, or other audio formats
                </p>
              </div>
            </div>
          )}
        </div>

        <button
          onClick={handleAnalyze}
          disabled={loading || !file}
          className="w-full bg-gradient-to-r from-green-500 to-emerald-500 text-white py-3 px-6 
                   rounded-xl font-semibold flex items-center justify-center gap-2
                   hover:shadow-lg hover:shadow-green-500/50 transition-all duration-300
                   disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing Voice...
            </>
          ) : (
            <>
              <Mic className="w-5 h-5" />
              Analyze Voice Emotion
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
            <div>
              <p className="text-red-300">{error}</p>
              {result?.suggestion && (
                <p className="text-red-300/80 text-sm mt-1">{result.suggestion}</p>
              )}
            </div>
          </motion.div>
        )}

        {result && !result.error && result.emotion && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 space-y-6"
          >
            <div className={`bg-gradient-to-r ${getEmotionColor(result.emotion)} p-6 rounded-xl`}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-3xl font-bold text-white mb-1">
                    {result.emotion}
                  </h3>
                  <p className="text-white/80">Detected Emotion</p>
                </div>
                <Mic className="w-12 h-12 text-white" />
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

            {/* All Emotions Breakdown */}
            {result.all_emotions && (
              <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                <h4 className="text-lg font-semibold mb-4 text-gray-200">Emotion Breakdown</h4>
                <div className="space-y-3">
                  {Object.entries(result.all_emotions)
                    .sort(([, a], [, b]) => b - a)
                    .map(([emotion, score]) => (
                      <div key={emotion}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-gray-300 capitalize">{emotion}</span>
                          <span className="text-gray-400 text-sm">{score}%</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-2">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${score}%` }}
                            transition={{ duration: 0.8, delay: 0.2 }}
                            className="bg-green-400 h-2 rounded-full"
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {/* Audio Features */}
            {result.audio_features && !result.audio_features.error && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <p className="text-gray-400 text-sm mb-1">Duration</p>
                  <p className="text-2xl font-bold text-white">{result.duration}s</p>
                </div>
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <p className="text-gray-400 text-sm mb-1">Energy Level</p>
                  <p className="text-2xl font-bold text-white">{result.audio_features.energy}</p>
                </div>
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <p className="text-gray-400 text-sm mb-1">Tempo</p>
                  <p className="text-2xl font-bold text-white">{result.audio_features.tempo} BPM</p>
                </div>
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <p className="text-gray-400 text-sm mb-1">Sample Rate</p>
                  <p className="text-2xl font-bold text-white">{result.audio_features.sample_rate} Hz</p>
                </div>
              </div>
            )}

            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h4 className="text-lg font-semibold mb-3 text-gray-200">Model Information</h4>
              <div className="space-y-2 text-gray-300">
                <p><span className="font-medium">Model:</span> {result.model}</p>
                <p className="text-sm text-gray-400 mt-4">{result.analysis}</p>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default VoiceAnalysis;
