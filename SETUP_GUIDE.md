# Setup Guide - AI Multi-Modal Analysis Platform

## Quick Start Guide

### Step 1: Clone/Download the Project
```bash
cd /path/to/windsurf-project
```

### Step 2: Backend Setup

#### 2.1 Create Virtual Environment
```bash
cd backend
python3 -m venv venv
```

#### 2.2 Activate Virtual Environment
**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

#### 2.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: This will download several packages. It may take 5-10 minutes depending on your internet speed.

#### 2.4 Start Backend Server
```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Keep this terminal open!**

### Step 3: Frontend Setup

Open a **new terminal window/tab**:

#### 3.1 Navigate to Frontend
```bash
cd frontend
```

#### 3.2 Install Dependencies
```bash
npm install
```

**Note**: This will download Node packages. It may take 2-3 minutes.

#### 3.3 Start Frontend Server
```bash
npm run dev
```

You should see:
```
VITE v5.0.8  ready in 500 ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

### Step 4: Access the Application

Open your browser and go to:
```
http://localhost:3000
```

## First-Time Model Downloads

When you use each feature for the first time, the models will be downloaded automatically:

1. **Sentiment Analysis** (~250MB) - DistilBERT model
2. **Text Summarization** (~1.6GB) - BART-Large model
3. **Face Analysis** (~100MB) - DeepFace models
4. **Voice Analysis** (~1.2GB) - Wav2Vec2 model

**Total Storage**: ~3-4GB for all models

The models are cached, so they only download once.

## Testing Each Feature

### 1. Sentiment Analysis
- Click "Sentiment Analysis" tab
- Enter text: "I love this amazing product!"
- Click "Analyze Sentiment"
- Wait 2-5 seconds for result

### 2. Text Summarization
- Click "Text Summarization" tab
- Click "Use sample text" or paste your own (50+ words)
- Click "Summarize Text"
- Wait 5-10 seconds for result

### 3. Face Analysis
- Click "Face Analysis" tab
- Upload a photo with a clear face
- Click "Analyze Face"
- Wait 3-7 seconds for result

### 4. Voice Analysis
- Click "Voice Analysis" tab
- Upload an audio file (WAV, MP3)
- Click "Analyze Voice Emotion"
- Wait 5-10 seconds for result

## Troubleshooting

### Backend Issues

#### Issue: "Module not found" errors
**Solution**:
```bash
cd backend
source venv/bin/activate  # Activate venv first!
pip install -r requirements.txt
```

#### Issue: "Port 8000 already in use"
**Solution**:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or change port in main.py:
# uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
```

#### Issue: Models downloading slowly
**Solution**: Be patient! Large models take time. Check your internet connection.

#### Issue: Out of memory errors
**Solution**: 
- Close other applications
- Use smaller models (edit model names in code)
- Run on a machine with more RAM (8GB+ recommended)

### Frontend Issues

#### Issue: "npm: command not found"
**Solution**: Install Node.js from https://nodejs.org/

#### Issue: Port 3000 already in use
**Solution**:
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or Vite will automatically use port 3001
```

#### Issue: "Failed to fetch" errors
**Solution**:
- Make sure backend is running on port 8000
- Check CORS settings in backend/main.py
- Try accessing http://localhost:8000/health in browser

### Model Issues

#### Issue: Face detection fails
**Solution**:
- Use a clear, well-lit photo
- Ensure face is visible and not too small
- Try a different image format (JPG, PNG)

#### Issue: Voice analysis fails
**Solution**:
- Convert audio to WAV format if possible
- Ensure audio is clear and not too noisy
- Check file size (under 10MB recommended)

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 5GB free space
- **OS**: macOS, Linux, or Windows 10+
- **Python**: 3.9+
- **Node.js**: 16+

### Recommended Requirements
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with CUDA (optional, for faster inference)

## GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Models will automatically use GPU if available
```

## Development Mode

### Backend Hot Reload
The backend automatically reloads when you change Python files.

### Frontend Hot Reload
The frontend automatically reloads when you change React files.

## Production Deployment

### Backend
```bash
cd backend
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
cd frontend
npm run build
# Deploy the 'dist' folder to your hosting service
```

## API Testing

You can test the API directly:

### Using cURL
```bash
# Test sentiment analysis
curl -X POST http://localhost:8000/api/sentiment \
  -F "text=I love this product!"

# Test health endpoint
curl http://localhost:8000/health
```

### Using Postman
1. Import the API endpoints
2. Set method to POST
3. Add form-data body
4. Send request

## Getting Help

### Check Logs
- **Backend logs**: Check the terminal where you ran `python main.py`
- **Frontend logs**: Check browser console (F12 → Console tab)

### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "Connection refused" | Backend not running | Start backend server |
| "CORS error" | Cross-origin issue | Check CORS settings |
| "Model not found" | Model not downloaded | Wait for download to complete |
| "Out of memory" | Not enough RAM | Close other apps or use smaller models |

## Next Steps

1. ✅ Complete the QUESTIONNAIRE.md
2. ✅ Test all four features
3. ✅ Take screenshots of results
4. ✅ Read the README.md for technical details
5. ✅ Customize and improve the models

## Support

For issues or questions:
1. Check this guide
2. Read README.md
3. Check error messages in terminal/console
4. Review the code comments

---

**Happy Coding! 🚀**
