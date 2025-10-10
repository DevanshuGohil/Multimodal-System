import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, Loader2, AlertCircle, User, Smile, Calendar } from 'lucide-react';
import axios from 'axios';

const FaceAnalysis = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setError(null);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select an image file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/api/face-analysis', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Failed to analyze face');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setError(null);
      setResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20"
      >
        <h2 className="text-3xl font-bold mb-6 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
          Face Analysis with DeepFace
        </h2>
        
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          className="mb-6 border-2 border-dashed border-white/30 rounded-xl p-8 text-center 
                   cursor-pointer hover:border-orange-400 hover:bg-white/5 transition-all duration-300"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          
          {preview ? (
            <div className="space-y-4">
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg shadow-lg"
              />
              <p className="text-gray-300">Click to change image</p>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="w-16 h-16 mx-auto text-gray-400" />
              <div>
                <p className="text-lg text-gray-300 mb-2">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-gray-500">
                  PNG, JPG, JPEG up to 10MB
                </p>
              </div>
            </div>
          )}
        </div>

        <button
          onClick={handleAnalyze}
          disabled={loading || !file}
          className="w-full bg-gradient-to-r from-orange-500 to-red-500 text-white py-3 px-6 
                   rounded-xl font-semibold flex items-center justify-center gap-2
                   hover:shadow-lg hover:shadow-orange-500/50 transition-all duration-300
                   disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing Face...
            </>
          ) : (
            <>
              <Smile className="w-5 h-5" />
              Analyze Face
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

        {result && !result.error && result.faces_detected > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 space-y-6"
          >
            <div className="bg-gradient-to-r from-orange-500 to-red-500 p-6 rounded-xl">
              <h3 className="text-2xl font-bold text-white mb-2">
                {result.faces_detected} Face{result.faces_detected > 1 ? 's' : ''} Detected
              </h3>
              <p className="text-white/80">{result.analysis}</p>
            </div>

            {result.faces.map((face, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white/5 rounded-xl p-6 border border-white/10"
              >
                {result.faces_detected > 1 && (
                  <h4 className="text-xl font-semibold mb-4 text-gray-200">
                    Face {face.face_number}
                  </h4>
                )}
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Age */}
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <Calendar className="w-5 h-5 text-orange-400" />
                      <h5 className="font-semibold text-gray-200">Age</h5>
                    </div>
                    <p className="text-3xl font-bold text-white">{face.age} years</p>
                  </div>

                  {/* Gender */}
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <User className="w-5 h-5 text-orange-400" />
                      <h5 className="font-semibold text-gray-200">Gender</h5>
                    </div>
                    <p className="text-2xl font-bold text-white mb-2">
                      {face.gender.prediction}
                    </p>
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div
                        style={{ width: `${face.gender.confidence}%` }}
                        className="bg-orange-400 h-2 rounded-full"
                      />
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                      {face.gender.confidence}% confidence
                    </p>
                  </div>

                  {/* Emotion */}
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <Smile className="w-5 h-5 text-orange-400" />
                      <h5 className="font-semibold text-gray-200">Emotion</h5>
                    </div>
                    <p className="text-2xl font-bold text-white mb-2">
                      {face.emotion.prediction}
                    </p>
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div
                        style={{ width: `${face.emotion.confidence}%` }}
                        className="bg-orange-400 h-2 rounded-full"
                      />
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                      {face.emotion.confidence}% confidence
                    </p>
                  </div>

                  {/* Race */}
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <User className="w-5 h-5 text-orange-400" />
                      <h5 className="font-semibold text-gray-200">Ethnicity</h5>
                    </div>
                    <p className="text-2xl font-bold text-white mb-2">
                      {face.race.prediction}
                    </p>
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div
                        style={{ width: `${face.race.confidence}%` }}
                        className="bg-orange-400 h-2 rounded-full"
                      />
                    </div>
                    <p className="text-sm text-gray-400 mt-1">
                      {face.race.confidence}% confidence
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}

            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h4 className="text-lg font-semibold mb-3 text-gray-200">Model Information</h4>
              <p className="text-gray-300">
                <span className="font-medium">Model:</span> {result.model}
              </p>
              <p className="text-sm text-gray-400 mt-2">
                Uses multiple deep learning models including VGG-Face, Facenet, and CNN-based architectures
                for comprehensive facial analysis.
              </p>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default FaceAnalysis;
