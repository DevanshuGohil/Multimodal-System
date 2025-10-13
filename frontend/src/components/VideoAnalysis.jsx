import React, { useState, useRef, useEffect } from 'react';
import {
  Video,
  Loader2,
  AlertCircle,
  BarChart2,
  FileVideo,
  Mic,
  User,
  Clock,
  Activity,
  CheckCircle,
  Brain,
  TrendingUp,
  Volume2,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { API_ENDPOINTS } from '../config/api';

const VideoAnalysis = () => {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [videoPreview, setVideoPreview] = useState(null);
  const [previewError, setPreviewError] = useState(false);
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState({
    status: 'idle',
    message: '',
    percentage: 0,
    currentStep: 0,
    totalSteps: 5,
  });

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  // Cleanup video URL on unmount
  useEffect(() => {
    return () => {
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview);
      }
    };
  }, [videoPreview]);

  useEffect(() => {
    if (progress.status === 'uploading') {
      setProgress(prev => ({
        ...prev,
        message: 'Uploading video...',
        currentStep: 1,
        percentage: 10
      }));
    } else if (progress.status === 'processing') {
      setProgress(prev => ({
        ...prev,
        message: 'Analyzing video content...',
        currentStep: 2,
        percentage: 30
      }));
    } else if (progress.status === 'processing_audio') {
      setProgress(prev => ({
        ...prev,
        message: 'Processing audio...',
        currentStep: 3,
        percentage: 50
      }));
    } else if (progress.status === 'analyzing_faces') {
      setProgress(prev => ({
        ...prev,
        message: 'Analyzing facial expressions...',
        currentStep: 4,
        percentage: 70
      }));
    } else if (progress.status === 'finalizing') {
      setProgress(prev => ({
        ...prev,
        message: 'Finalizing results...',
        currentStep: 5,
        percentage: 90
      }));
    } else if (progress.status === 'completed') {
      setProgress(prev => ({
        ...prev,
        message: 'Analysis complete!',
        percentage: 100
      }));
    }
  }, [progress.status]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 100 * 1024 * 1024) {
        setError('File size should be less than 100MB');
        return;
      }

      // Revoke old URL if exists
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview);
      }

      setFile(selectedFile);
      setError('');
      setPreviewError(false);
      setResults(null);

      // Create new preview URL
      try {
        const videoUrl = URL.createObjectURL(selectedFile);
        setVideoPreview(videoUrl);

        // Set video source after a brief delay to ensure element is ready
        setTimeout(() => {
          if (videoRef.current) {
            videoRef.current.load();
          }
        }, 100);
      } catch (err) {
        console.error('Error creating video preview:', err);
        setPreviewError(true);
      }
    }
  };

  const handleRemoveFile = () => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setFile(null);
    setVideoPreview(null);
    setPreviewError(false);
    setError('');
    setResults(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setIsLoading(true);
    setError('');
    setProgress({
      status: 'uploading',
      message: 'Uploading video...',
      percentage: 0,
      currentStep: 1,
      totalSteps: 5
    });

    try {
      const response = await axios.post(API_ENDPOINTS.videoAnalysis, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000,
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          );
          setProgress(prev => ({
            ...prev,
            status: 'uploading',
            percentage: Math.min(20, percentCompleted * 0.2)
          }));
        },
      });

      // Simulate processing steps for better UX
      setProgress(prev => ({ ...prev, status: 'processing', percentage: 30 }));
      await new Promise(resolve => setTimeout(resolve, 500));

      setProgress(prev => ({ ...prev, status: 'processing_audio', percentage: 50 }));
      await new Promise(resolve => setTimeout(resolve, 500));

      setProgress(prev => ({ ...prev, status: 'analyzing_faces', percentage: 70 }));
      await new Promise(resolve => setTimeout(resolve, 500));

      setProgress(prev => ({ ...prev, status: 'finalizing', percentage: 90 }));

      if (response.data.status === 'success') {
        setResults(response.data);
        setProgress({
          status: 'completed',
          message: 'Analysis complete!',
          percentage: 100,
          currentStep: 5,
          totalSteps: 5
        });
      } else {
        throw new Error(response.data.error || 'Analysis failed');
      }

    } catch (err) {
      console.error('Error analyzing video:', err);
      setError(err.response?.data?.error || err.message || 'Failed to analyze video. Please try again.');
      setProgress(prev => ({
        ...prev,
        status: 'error',
        message: 'Analysis failed',
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      happiness: 'from-yellow-400 to-orange-400',
      joy: 'from-yellow-400 to-orange-400',
      sadness: 'from-blue-400 to-indigo-400',
      anger: 'from-red-400 to-pink-400',
      fear: 'from-purple-400 to-indigo-400',
      surprise: 'from-pink-400 to-purple-400',
      disgust: 'from-green-400 to-emerald-400',
      neutral: 'from-gray-400 to-slate-400'
    };
    return colors[emotion.toLowerCase()] || 'from-gray-400 to-slate-400';
  };

  const getSincerityBadge = (score) => {
    if (score >= 0.8) return { color: 'from-green-500 to-emerald-500', label: 'High' };
    if (score >= 0.6) return { color: 'from-blue-500 to-cyan-500', label: 'Moderate' };
    if (score >= 0.4) return { color: 'from-yellow-500 to-orange-500', label: 'Low' };
    return { color: 'from-red-500 to-pink-500', label: 'Very Low' };
  };

  const renderEmotionBars = (emotions) => {
    if (!emotions || typeof emotions !== 'object') return null;

    return (
      <div className="space-y-3">
        {Object.entries(emotions)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .map(([emotion, score]) => (
            <div key={emotion} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="capitalize font-medium text-gray-300">{emotion}</span>
                <span className="text-gray-400">{(score * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2.5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${score * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className={`bg-gradient-to-r ${getEmotionColor(emotion)} h-2.5 rounded-full`}
                />
              </div>
            </div>
          ))}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden"
      >
        <div className="p-8">
          <div className="flex items-center mb-6">
            <Video className="w-8 h-8 text-cyan-400 mr-3" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Video Analysis
            </h1>
          </div>

          <p className="text-gray-300 text-sm mb-6">
            Upload a video to analyze facial expressions, speech content, and voice characteristics.
            The analysis includes emotion detection, speech-to-text, and sincerity scoring.
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-white/20 rounded-xl cursor-pointer bg-white/5 hover:bg-white/10 transition-all duration-300">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <FileVideo className="w-12 h-12 text-gray-400 mb-3" />
                  <p className="mb-2 text-sm text-gray-300">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">MP4, AVI, or MOV (max. 100MB)</p>
                  {file && (
                    <p className="mt-2 text-sm text-cyan-400 font-medium">
                      Selected: {file.name}
                    </p>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept="video/*"
                  onChange={handleFileChange}
                />
              </label>
            </div>

            {file && (
              <div className="mt-4">
                <div className="relative">
                  {/* Video Preview or File Info */}
                  {!previewError && videoPreview ? (
                    <div className="relative">
                      <video
                        ref={videoRef}
                        className="w-full rounded-xl border border-white/20 max-h-96 bg-black"
                        controls
                        playsInline
                        preload="metadata"
                        onError={(e) => {
                          console.error('Video playback error:', e);
                          setPreviewError(true);
                        }}
                      >
                        <source src={videoPreview} type={file.type} />
                        Your browser does not support the video tag.
                      </video>

                      <button
                        type="button"
                        onClick={handleRemoveFile}
                        className="absolute top-2 right-2 bg-red-500/80 hover:bg-red-600 text-white p-2 rounded-lg transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    // Fallback: Show file info card
                    <div className="relative p-6 bg-white/5 rounded-xl border border-white/10">
                      <div className="flex items-start gap-4">
                        <FileVideo className="w-12 h-12 text-cyan-400 flex-shrink-0" />
                        <div className="flex-1">
                          <h3 className="text-white font-medium mb-1">{file.name}</h3>
                          <p className="text-gray-400 text-sm">
                            Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                          <p className="text-gray-400 text-sm">
                            Type: {file.type || 'Unknown'}
                          </p>
                          {previewError && (
                            <p className="text-yellow-400 text-xs mt-2">
                              ⚠️ Preview unavailable, but file can still be analyzed
                            </p>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={handleRemoveFile}
                          className="bg-red-500/80 hover:bg-red-600 text-white p-2 rounded-lg transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Loading Overlay */}
                  <AnimatePresence>
                    {(isLoading || progress.percentage > 0) && progress.status !== 'completed' && (
                      <motion.div
                        className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center rounded-xl p-6 text-center"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                      >
                        <div className="relative w-full max-w-md bg-white/10 rounded-full h-4 mb-6 overflow-hidden">
                          <motion.div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300"
                            initial={{ width: '0%' }}
                            animate={{ width: `${progress.percentage}%` }}
                            transition={{ duration: 0.5 }}
                          />
                          <div className="absolute inset-0 flex items-center justify-center text-xs text-white font-semibold">
                            {progress.percentage}%
                          </div>
                        </div>

                        <div className="flex items-center space-x-3 mb-4">
                          {progress.status === 'completed' ? (
                            <CheckCircle className="w-6 h-6 text-green-400" />
                          ) : (
                            <Activity className="w-6 h-6 text-cyan-400 animate-pulse" />
                          )}
                          <h3 className="text-lg font-medium text-white">
                            {progress.status === 'completed' ? 'Analysis Complete!' : progress.message}
                          </h3>
                        </div>

                        <div className="text-sm text-gray-300">
                          Step {progress.currentStep} of {progress.totalSteps}
                        </div>

                        {progress.status !== 'completed' && (
                          <div className="mt-4 text-xs text-gray-400">
                            This may take a few minutes. Please don't close this window.
                          </div>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Error Message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-xl flex items-start gap-3"
                  >
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-red-300">{error}</p>
                  </motion.div>
                )}
              </div>
            )}

            <div className="flex justify-end mt-6">
              <button
                type="submit"
                disabled={isLoading || !file}
                className={`px-6 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-blue-500 to-cyan-500
                         hover:shadow-lg hover:shadow-cyan-500/50 transition-all duration-300
                         disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 flex items-center gap-2
                         ${(isLoading || !file) ? '' : ''}`}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Video className="w-5 h-5" />
                    Analyze Video
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </motion.div>

      {/* Results Section */}
      {results && results.status === 'success' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 space-y-8"
        >
          {/* Video Info Summary */}
          <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 p-6">
            <h2 className="text-xl font-bold mb-4 text-white">Video Information</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                <div className="text-sm text-gray-400">Duration</div>
                <div className="text-2xl font-bold text-white">{results.video_info.duration_seconds.toFixed(1)}s</div>
              </div>
              <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                <div className="text-sm text-gray-400">FPS</div>
                <div className="text-2xl font-bold text-white">{results.video_info.fps}</div>
              </div>
              <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                <div className="text-sm text-gray-400">Resolution</div>
                <div className="text-2xl font-bold text-white">{results.video_info.resolution}</div>
              </div>
              <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                <div className="text-sm text-gray-400">Frames Analyzed</div>
                <div className="text-2xl font-bold text-white">{results.video_info.frames_analyzed}</div>
              </div>
            </div>
          </div>

          {/* Sincerity Score - Featured */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden">
            <div className="p-6 border-b border-white/10 bg-gradient-to-r from-green-500/10 to-blue-500/10">
              <h2 className="text-lg font-semibold text-white flex items-center">
                <Brain className="w-5 h-5 mr-2 text-cyan-400" />
                Sincerity & Authenticity Analysis
              </h2>
            </div>
            <div className="p-6">
              <div className="text-center mb-6">
                <div className={`inline-flex items-center justify-center w-32 h-32 rounded-full bg-gradient-to-r ${getSincerityBadge(results.sincerity.overall_score).color} text-white mb-4`}>
                  <div className="text-4xl font-bold">
                    {(results.sincerity.overall_score * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="text-xl font-semibold text-white mb-2">
                  {results.sincerity.interpretation}
                </div>
                <span className={`inline-block px-4 py-1 rounded-full text-sm font-medium bg-gradient-to-r ${getSincerityBadge(results.sincerity.overall_score).color} text-white`}>
                  {getSincerityBadge(results.sincerity.overall_score).label} Authenticity
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-2">Emotion Congruence</div>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold text-cyan-400">
                      {(results.sincerity.emotion_congruence * 100).toFixed(1)}%
                    </div>
                    <TrendingUp className="w-6 h-6 text-cyan-400" />
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${results.sincerity.emotion_congruence * 100}%` }}
                      transition={{ duration: 1 }}
                      className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full"
                    />
                  </div>
                </div>

                <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-2">Speech Quality</div>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold text-green-400">
                      {(results.sincerity.speech_quality * 100).toFixed(1)}%
                    </div>
                    <Mic className="w-6 h-6 text-green-400" />
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${results.sincerity.speech_quality * 100}%` }}
                      transition={{ duration: 1 }}
                      className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full"
                    />
                  </div>
                </div>

                <div className="bg-white/5 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-2">Voice Consistency</div>
                  <div className="flex items-center justify-between">
                    <div className="text-2xl font-bold text-purple-400">
                      {(results.sincerity.voice_consistency * 100).toFixed(1)}%
                    </div>
                    <Volume2 className="w-6 h-6 text-purple-400" />
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${results.sincerity.voice_consistency * 100}%` }}
                      transition={{ duration: 1 }}
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-500/10 rounded-xl border border-blue-500/20">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">Filler Words Detected:</span>
                  <span className="font-semibold text-white">
                    {results.sincerity.filler_word_count} ({(results.sincerity.filler_ratio * 100).toFixed(1)}% of speech)
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Transcript Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden">
            <div className="p-6 border-b border-white/10">
              <h2 className="text-lg font-semibold text-white flex items-center">
                <Mic className="w-5 h-5 mr-2 text-cyan-400" />
                Transcript
                <span className="ml-auto text-sm text-gray-400">
                  {results.transcript.word_count} words • {results.transcript.character_count} characters
                </span>
              </h2>
            </div>
            <div className="p-6">
              <div className="bg-white/5 rounded-xl p-4 max-h-96 overflow-y-auto border border-white/10">
                <p className="whitespace-pre-line text-gray-300 leading-relaxed">
                  {results.transcript.text || 'No speech detected in the video.'}
                </p>
              </div>
            </div>
          </div>

          {/* Emotions Analysis Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Facial Emotions */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden">
              <div className="p-6 border-b border-white/10">
                <h2 className="text-lg font-semibold text-white flex items-center">
                  <User className="w-5 h-5 mr-2 text-cyan-400" />
                  Facial Emotion Analysis
                </h2>
              </div>
              <div className="p-6 space-y-6">
                <div className="text-center">
                  <div className={`inline-block px-6 py-3 rounded-full bg-gradient-to-r ${getEmotionColor(results.emotions.face.dominant)} text-white text-xl font-bold mb-2`}>
                    {results.emotions.face.dominant.toUpperCase()}
                  </div>
                  <div className="text-sm text-gray-400 mt-2">
                    Confidence: {(results.emotions.face.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div>
                  <h3 className="font-medium text-gray-200 mb-3">All Detected Emotions</h3>
                  {renderEmotionBars(results.emotions.face.all_emotions)}
                </div>
              </div>
            </div>

            {/* Text Emotions */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden">
              <div className="p-6 border-b border-white/10">
                <h2 className="text-lg font-semibold text-white flex items-center">
                  <BarChart2 className="w-5 h-5 mr-2 text-cyan-400" />
                  Text Sentiment Analysis
                </h2>
              </div>
              <div className="p-6 space-y-6">
                <div className="text-center">
                  <div className={`inline-block px-6 py-3 rounded-full bg-gradient-to-r ${getEmotionColor(results.emotions.text.dominant)} text-white text-xl font-bold mb-2`}>
                    {results.emotions.text.dominant.toUpperCase()}
                  </div>
                  <div className="text-sm text-gray-400 mt-2">
                    Confidence: {(results.emotions.text.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div>
                  <h3 className="font-medium text-gray-200 mb-3">Top Emotions from Text</h3>
                  <div className="space-y-3">
                    {results.emotions.text.top_emotions.map((emotion, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-xl border border-white/10">
                        <span className="capitalize font-medium text-gray-300">{emotion.emotion}</span>
                        <span className="text-sm text-gray-400">{(emotion.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Audio Features */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden">
            <div className="p-6 border-b border-white/10">
              <h2 className="text-lg font-semibold text-white flex items-center">
                <Volume2 className="w-5 h-5 mr-2 text-cyan-400" />
                Audio Analysis
              </h2>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-blue-500/20 to-indigo-500/20 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-1">Mean Volume</div>
                  <div className="text-2xl font-bold text-blue-400">{results.audio_features.mean_volume_db} dB</div>
                </div>
                <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-1">Max Volume</div>
                  <div className="text-2xl font-bold text-purple-400">{results.audio_features.max_volume_db} dB</div>
                </div>
                <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-1">RMS Energy</div>
                  <div className="text-2xl font-bold text-green-400">{results.audio_features.rms_energy.toFixed(3)}</div>
                </div>
                <div className="bg-gradient-to-br from-yellow-500/20 to-orange-500/20 p-4 rounded-xl border border-white/10">
                  <div className="text-sm text-gray-400 mb-1">Energy Level</div>
                  <div className="text-2xl font-bold text-orange-400 capitalize">{results.audio_features.energy_level}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 overflow-hidden">
            <div className="p-6 border-b border-white/10">
              <h2 className="text-lg font-semibold text-white flex items-center">
                <Clock className="w-5 h-5 mr-2 text-cyan-400" />
                Analysis Metadata
              </h2>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium text-gray-200 mb-3">Analyzed At</h3>
                  <p className="text-gray-400">{new Date(results.metadata.analyzed_at).toLocaleString()}</p>
                </div>
                <div>
                  <h3 className="font-medium text-gray-200 mb-3">Model Versions</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Speech-to-Text:</span>
                      <span className="font-mono text-gray-300">{results.metadata.model_versions.speech_to_text}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Face Emotion:</span>
                      <span className="font-mono text-gray-300">{results.metadata.model_versions.face_emotion}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Text Sentiment:</span>
                      <span className="font-mono text-gray-300 text-xs">{results.metadata.model_versions.text_sentiment}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default VideoAnalysis;
