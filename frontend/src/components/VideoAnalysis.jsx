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
  CheckCircle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { API_ENDPOINTS } from '../config/api';

const VideoAnalysis = () => {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState({
    status: 'idle', // 'idle' | 'uploading' | 'processing' | 'completed' | 'error'
    message: '',
    percentage: 0,
    currentStep: 0,
    totalSteps: 5, // Total number of steps in the process
  });

  // Update progress message based on status
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
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 100 * 1024 * 1024) { // 100MB limit
        setError('File size should be less than 100MB');
        return;
      }
      
      // Revoke previous object URL to prevent memory leaks
      if (file) {
        URL.revokeObjectURL(URL.createObjectURL(file));
      }
      
      setFile(selectedFile);
      setError('');
      setResults(null);
      
      // Create preview URL for the video
      const videoUrl = URL.createObjectURL(selectedFile);
      if (videoRef.current) {
        videoRef.current.src = videoUrl;
        videoRef.current.load(); // Ensure the new source is loaded
        videoRef.current.play().catch(error => {
          console.error('Error playing video:', error);
          setError('Could not play video. The file may be corrupted or in an unsupported format.');
        });
      }
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
    
    let eventSource = null;
    
    try {
      // First, start the analysis and get an analysis ID
      const startResponse = await axios.post(API_ENDPOINTS.videoAnalysis, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          );
          setProgress(prev => ({
            ...prev,
            percentage: Math.min(20, percentCompleted) // Cap at 20% for upload
          }));
        },
      });
      
      // After upload, connect to progress endpoint
      const analysisId = startResponse.data.analysis_id || 'default';
      eventSource = new EventSource(`${API_ENDPOINTS.videoAnalysis}/progress/${analysisId}`);
      
      return new Promise((resolve, reject) => {
        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            setProgress(prev => ({
              ...prev,
              status: data.status || prev.status,
              message: data.message || prev.message,
              percentage: data.percentage || prev.percentage,
              currentStep: data.step || prev.currentStep
            }));
            
            // If analysis is complete, resolve the promise
            if (data.status === 'completed') {
              setResults(data.result || {});
              if (eventSource) eventSource.close();
              resolve(data);
            }
          } catch (err) {
            console.error('Error processing progress update:', err);
          }
        };

        eventSource.onerror = (error) => {
          console.error('EventSource error:', error);
          if (eventSource) eventSource.close();
          reject(new Error('Connection to progress updates failed'));
        };
      });
    } catch (err) {
      console.error('Error analyzing video:', err);
      setError(err.response?.data?.detail || 'Failed to analyze video. Please try again.');
      setProgress(prev => ({
        ...prev,
        status: 'error',
        message: 'Analysis failed',
      }));
      throw err;
    } finally {
      setIsLoading(false);
      if (eventSource) {
        eventSource.close();
      }
    }
  };

  const renderEmotionBars = (emotions) => {
    if (!emotions || !Array.isArray(emotions) || emotions.length === 0) return null;
    
    // Get the first emotion object
    const emotionData = emotions[0];
    
    return (
      <div className="space-y-2">
        {Object.entries(emotionData).map(([emotion, score]) => (
          <div key={emotion} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="capitalize font-medium">{emotion}</span>
              <span className="text-gray-500">{(score * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-indigo-600 h-2.5 rounded-full" 
                style={{ width: `${score * 100}%` }}
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
        className="bg-white rounded-xl shadow-md overflow-hidden"
      >
        <div className="p-6">
          <div className="flex items-center mb-6">
            <Video className="w-8 h-8 text-indigo-600 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900">Video Analysis</h1>
          </div>
          
          <p className="text-gray-600 mb-6">
            Upload a video to analyze facial expressions, speech content, and voice characteristics.
            The analysis includes emotion detection, speech-to-text, and audio feature extraction.
          </p>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <FileVideo className="w-12 h-12 text-gray-400 mb-3" />
                  <p className="mb-2 text-sm text-gray-500">
                    <span className="font-semibold">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500">MP4, AVI, or MOV (max. 100MB)</p>
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
                  <video 
                    ref={videoRef}
                    className="w-full rounded-lg border border-gray-200"
                    controls
                    playsInline
                    onCanPlay={() => {
                      if (videoRef.current) {
                        videoRef.current.play().catch(error => {
                          console.error('Playback failed:', error);
                          setError('Autoplay was prevented. Please click the play button to start the video.');
                        });
                      }
                    }}
                    onError={(e) => {
                      console.error('Video error:', e);
                      setError('Error loading video. Please try another file.');
                    }}
                  >
                    <source src={URL.createObjectURL(file)} type={file.type} />
                    Your browser does not support the video tag.
                  </video>
                  
                  <AnimatePresence>
                    {(isLoading || progress.percentage > 0) && (
                      <motion.div 
                        className="absolute inset-0 bg-black bg-opacity-70 flex flex-col items-center justify-center rounded-lg p-6 text-center"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                      >
                        <div className="relative w-full max-w-md bg-gray-800 rounded-full h-4 mb-6 overflow-hidden">
                          <motion.div 
                            className="h-full bg-indigo-600 transition-all duration-300"
                            initial={{ width: '0%' }}
                            animate={{ width: `${progress.percentage}%` }}
                            transition={{ duration: 0.5 }}
                          />
                          <div className="absolute inset-0 flex items-center justify-center text-xs text-white">
                            {progress.percentage}%
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-3 mb-4">
                          {progress.status === 'completed' ? (
                            <CheckCircle className="w-6 h-6 text-green-500" />
                          ) : (
                            <Activity className="w-6 h-6 text-indigo-400 animate-pulse" />
                          )}
                          <h3 className="lg font-medium text-white">
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
                
                {error && (
                  <div className="mt-4 p-4 text-sm text-red-700 bg-red-100 rounded-lg flex items-center">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    {error}
                  </div>
                )}
              </div>
            )}
            
            <div className="flex justify-end mt-6">
              <button
                type="submit"
                disabled={isLoading || !file}
                className={`px-6 py-3 rounded-lg font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors ${
                  (isLoading || !file) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <Loader2 className="animate-spin mr-2 h-5 w-5 text-white" />
                    Processing...
                  </div>
                ) : (
                  'Analyze Video'
                )}
              </button>
            </div>
          </form>
        </div>
      </motion.div>
      
      {results && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 space-y-8"
        >
          {/* Transcript Section */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <Mic className="w-5 h-5 mr-2 text-indigo-600" />
                Transcript
              </h2>
            </div>
            <div className="p-6">
              <p className="whitespace-pre-line text-gray-700">
                {results.transcript || 'No speech detected in the video.'}
              </p>
            </div>
          </div>
          
          {/* Facial Analysis Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white rounded-xl shadow-md overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <User className="w-5 h-5 mr-2 text-indigo-600" />
                  Facial Analysis
                </h2>
              </div>
              <div className="p-6 space-y-6">
                {results.facial_analysis.emotions.length > 0 ? (
                  <>
                    <div>
                      <h3 className="font-medium text-gray-900 mb-3">Dominant Emotion</h3>
                      <div className="bg-indigo-50 rounded-lg p-4">
                        <span className="text-indigo-800 font-medium capitalize">
                          {results.facial_analysis.dominant_emotion}
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-gray-900 mb-3">Age Statistics</h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-gray-50 p-3 rounded-lg text-center">
                          <div className="text-sm text-gray-500">Min</div>
                          <div className="text-lg font-semibold">{results.facial_analysis.age_stats.min?.toFixed(1) || 'N/A'}</div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg text-center">
                          <div className="text-sm text-gray-500">Avg</div>
                          <div className="text-lg font-semibold">{results.facial_analysis.age_stats.avg?.toFixed(1) || 'N/A'}</div>
                        </div>
                        <div className="bg-gray-50 p-3 rounded-lg text-center">
                          <div className="text-sm text-gray-500">Max</div>
                          <div className="text-lg font-semibold">{results.facial_analysis.age_stats.max?.toFixed(1) || 'N/A'}</div>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="font-medium text-gray-900 mb-3">Gender Distribution</h3>
                      <div className="flex space-x-4">
                        {Object.entries(results.facial_analysis.gender_distribution).map(([gender, count]) => (
                          <div key={gender} className="flex-1 bg-gray-50 p-3 rounded-lg text-center">
                            <div className="text-sm text-gray-500 capitalize">{gender}</div>
                            <div className="text-lg font-semibold">{count}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="text-gray-500 text-center py-4">No faces detected in the video.</p>
                )}
              </div>
            </div>
            
            {/* Audio Analysis Section */}
            <div className="bg-white rounded-xl shadow-md overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <BarChart2 className="w-5 h-5 mr-2 text-indigo-600" />
                  Audio Analysis
                </h2>
              </div>
              <div className="p-6 space-y-6">
                <div>
                  <h3 className="font-medium text-gray-900 mb-3">Audio Features</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Tempo</div>
                      <div className="text-lg font-semibold">{results.audio_analysis.tempo.toFixed(1)} BPM</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Energy</div>
                      <div className="text-lg font-semibold">{results.audio_analysis.energy.toFixed(4)}</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-500">Speech Rate</div>
                      <div className="text-lg font-semibold">{results.audio_analysis.speech_rate.toFixed(1)} words/sec</div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-gray-900 mb-3">Text Emotions</h3>
                  {results.text_emotions && results.text_emotions.length > 0 ? (
                    renderEmotionBars(results.text_emotions)
                  ) : (
                    <p className="text-gray-500">No emotion data available from text.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          {/* Metrics Section */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <Clock className="w-5 h-5 mr-2 text-indigo-600" />
                Processing Metrics
              </h2>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-500">Frames Analyzed</div>
                  <div className="text-2xl font-bold text-indigo-600">{results.metrics.frames_analyzed}</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-500">Faces Detected</div>
                  <div className="text-2xl font-bold text-indigo-600">{results.metrics.faces_detected}</div>
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
