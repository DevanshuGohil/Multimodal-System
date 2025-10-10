# ⚡ Quick Start - 2 Minutes to Running

## 🚀 Fastest Way to Start

### Option 1: Using Scripts (Recommended)

**Terminal 1 - Backend:**
```bash
cd /Users/devanshugohil/Documents/Voicera.io\ OA/CascadeProjects/windsurf-project
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd /Users/devanshugohil/Documents/Voicera.io\ OA/CascadeProjects/windsurf-project
./start_frontend.sh
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Access the App

Open browser: **http://localhost:3000**

## 🎯 Test Each Feature

1. **Sentiment Analysis** - Type: "I love this amazing product!"
2. **Text Summarization** - Click "Use sample text"
3. **Face Analysis** - Upload a photo with a face
4. **Voice Analysis** - Upload an audio file (WAV/MP3)

## 📦 What Gets Downloaded

First-time use downloads models (~3-4GB total):
- DistilBERT: ~250MB
- BART: ~1.6GB
- DeepFace: ~100MB
- Wav2Vec2: ~1.2GB

Models cache automatically - only download once!

## ❓ Problems?

**Backend won't start:**
```bash
# Check Python version
python3 --version  # Need 3.9+

# Try reinstalling
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend won't start:**
```bash
# Check Node version
node --version  # Need 16+

# Try reinstalling
cd frontend
rm -rf node_modules
npm install
```

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

## 📚 More Help

- **Detailed Setup**: See `SETUP_GUIDE.md`
- **Technical Details**: See `README.md`
- **Project Overview**: See `PROJECT_OVERVIEW.md`

---

**That's it! You're ready to go! 🎉**
