/**
 * API Configuration
 * Centralized API endpoint configuration for the application
 * 
 * Environment Variables:
 * - VITE_API_URL: Backend API URL (set in .env files)
 * 
 * Usage:
 * import { API_BASE_URL, API_ENDPOINTS } from '@/config/api';
 */

// Get API URL from environment variable or use default for development
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// API Endpoints
export const API_ENDPOINTS = {
  sentiment: `${API_BASE_URL}/api/sentiment`,
  summarize: `${API_BASE_URL}/api/summarize`,
  faceAnalysis: `${API_BASE_URL}/api/face-analysis`,
  voiceAnalysis: `${API_BASE_URL}/api/voice-analysis`,
  health: `${API_BASE_URL}/`,
};

// API Configuration
export const API_CONFIG = {
  timeout: 120000, // 2 minutes for ML model inference
  headers: {
    'Content-Type': 'application/json',
  },
};

// Helper function to check if API is available
export const checkAPIHealth = async () => {
  try {
    const response = await fetch(API_ENDPOINTS.health);
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};

export default {
  API_BASE_URL,
  API_ENDPOINTS,
  API_CONFIG,
  checkAPIHealth,
};
