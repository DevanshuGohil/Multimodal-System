import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, MessageSquare, FileText, Image, Mic } from 'lucide-react';
import Header from './components/Header';
import SentimentAnalysis from './components/SentimentAnalysis';
import TextSummarization from './components/TextSummarization';
import FaceAnalysis from './components/FaceAnalysis';
import VoiceAnalysis from './components/VoiceAnalysis';

function App() {
  const [activeTab, setActiveTab] = useState('sentiment');

  const tabs = [
    { id: 'sentiment', name: 'Sentiment Analysis', icon: MessageSquare, color: 'from-blue-500 to-cyan-500' },
    { id: 'summarize', name: 'Text Summarization', icon: FileText, color: 'from-purple-500 to-pink-500' },
    { id: 'face', name: 'Face Analysis', icon: Image, color: 'from-orange-500 to-red-500' },
    { id: 'voice', name: 'Voice Analysis', icon: Mic, color: 'from-green-500 to-emerald-500' },
  ];

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'sentiment':
        return <SentimentAnalysis />;
      case 'summarize':
        return <TextSummarization />;
      case 'face':
        return <FaceAnalysis />;
      case 'voice':
        return <VoiceAnalysis />;
      default:
        return <SentimentAnalysis />;
    }
  };

  return (
    <div className="min-h-screen text-white">
      <Header />
      
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-16 h-16 text-indigo-400 mr-4" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              AI Multi-Modal Analysis
            </h1>
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Powered by state-of-the-art Transformer models for text, image, and voice analysis
          </p>
        </motion.div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {tabs.map((tab, index) => {
            const Icon = tab.icon;
            return (
              <motion.button
                key={tab.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-3 px-6 py-3 rounded-xl font-semibold
                  transition-all duration-300 transform hover:scale-105
                  ${activeTab === tab.id
                    ? `bg-gradient-to-r ${tab.color} text-white shadow-lg shadow-${tab.color}/50`
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                {tab.name}
              </motion.button>
            );
          })}
        </div>

        {/* Active Component */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          {renderActiveComponent()}
        </motion.div>
      </div>

      {/* Footer */}
      <footer className="mt-16 py-8 border-t border-white/10">
        <div className="container mx-auto px-4 text-center text-gray-400">
          <p className="mb-2">Built with ❤️ using Transformer Models</p>
          <p className="text-sm">
            Models: RoBERTa • BART • DeepFace • Wav2Vec2
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
