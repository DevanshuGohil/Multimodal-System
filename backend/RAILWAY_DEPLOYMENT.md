# 🚂 Railway Deployment Guide

This guide will help you deploy your FastAPI backend to Railway.

## 📋 Prerequisites

1. A [Railway account](https://railway.app/) (sign up with GitHub)
2. Your backend code pushed to a GitHub repository
3. Railway CLI (optional, but recommended)

---

## 🚀 Method 1: Deploy via Railway Dashboard (Easiest)

### Step 1: Create a New Project

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub account
5. Select your repository: `windsurf-project`

### Step 2: Configure the Service

1. Railway will auto-detect your Python app
2. Click on your service to open settings
3. Go to **"Settings"** tab
4. Set the following:
   - **Root Directory**: `backend`
   - **Build Command**: (leave empty, Railway auto-detects)
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 3: Add Environment Variables

1. Go to **"Variables"** tab
2. Click **"+ New Variable"**
3. Add these variables:

```
TOKENIZERS_PARALLELISM=false
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### Step 4: Deploy

1. Railway will automatically deploy your app
2. Wait for the build to complete (this may take 5-10 minutes due to ML libraries)
3. Once deployed, you'll see a **"Deployments"** section with a green checkmark

### Step 5: Get Your Backend URL

1. Go to **"Settings"** tab
2. Click **"Generate Domain"** under **"Networking"**
3. Copy your Railway URL (e.g., `https://your-app.up.railway.app`)
4. Test it by visiting: `https://your-app.up.railway.app/` (should show API info)

---

## 🛠️ Method 2: Deploy via Railway CLI

### Step 1: Install Railway CLI

```bash
# macOS
brew install railway

# Or using npm
npm i -g @railway/cli
```

### Step 2: Login to Railway

```bash
railway login
```

### Step 3: Initialize Project

```bash
cd backend
railway init
```

Select "Create new project" and give it a name.

### Step 4: Deploy

```bash
railway up
```

### Step 5: Add Environment Variables

```bash
railway variables set TOKENIZERS_PARALLELISM=false
railway variables set ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### Step 6: Open Your App

```bash
railway open
```

---

## 🔧 Configuration Files Explained

### `railway.json`
Configures Railway build and deployment settings.

### `Procfile`
Tells Railway how to start your application.

### `requirements.txt`
Lists all Python dependencies. Railway automatically installs these.

### `runtime.txt`
Specifies Python version (3.11).

### `.railwayignore`
Files/folders to exclude from deployment (like `.gitignore`).

---

## 🌐 Update Frontend to Use Railway Backend

After deployment, update your frontend's API URL:

### Option 1: Environment Variable (Recommended)

Create/update `frontend/.env.production`:

```env
VITE_API_URL=https://your-app.up.railway.app
```

### Option 2: Direct Update

Update `frontend/src/config/api.js`:

```javascript
export const API_BASE_URL = 
  import.meta.env.VITE_API_URL || 
  'https://your-app.up.railway.app';
```

---

## 📊 Monitoring & Logs

### View Logs
1. Go to your Railway dashboard
2. Click on your service
3. Go to **"Deployments"** tab
4. Click on the latest deployment
5. View real-time logs

### Or via CLI:
```bash
railway logs
```

---

## ⚠️ Important Notes

### 1. **Build Time**
- First deployment takes 5-10 minutes due to ML libraries (PyTorch, TensorFlow)
- Subsequent deployments are faster (Railway caches dependencies)

### 2. **Memory Usage**
- Your app uses ML models which require significant RAM
- Railway's free tier provides 512MB RAM
- Consider upgrading to **Hobby plan ($5/month)** for 8GB RAM
- This is **essential** for video analysis to work properly

### 3. **Cold Starts**
- Railway may sleep your app after inactivity (free tier)
- First request after sleep takes longer (30-60 seconds)
- Hobby plan keeps your app always running

### 4. **File Storage**
- Temporary files are stored in `/tmp`
- Railway's filesystem is ephemeral (resets on redeploy)
- Video files are automatically cleaned up after processing

### 5. **CORS Configuration**
- Backend is configured to accept requests from Railway/Vercel/Netlify domains
- Add your custom domain to `ALLOWED_ORIGINS` environment variable

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Check logs
railway logs

# Common issues:
# - Out of memory: Upgrade to Hobby plan
# - Missing dependencies: Check requirements.txt
```

### App Crashes
```bash
# View crash logs
railway logs --tail 100

# Common fixes:
# - Increase memory (upgrade plan)
# - Check environment variables
# - Verify Python version compatibility
```

### CORS Errors
Add your frontend domain to environment variables:
```bash
railway variables set ALLOWED_ORIGINS=https://your-frontend.com
```

### Slow Response
- ML models load on first request (lazy loading)
- Consider keeping app warm with a health check ping
- Upgrade to Hobby plan for better performance

---

## 💰 Pricing

### Free Tier (Starter)
- ✅ 512MB RAM
- ✅ Shared CPU
- ✅ $5 free credit/month
- ⚠️ May not be enough for video analysis

### Hobby Plan ($5/month)
- ✅ 8GB RAM
- ✅ Shared CPU
- ✅ No sleep
- ✅ **Recommended for this app**

### Pro Plan ($20/month)
- ✅ 32GB RAM
- ✅ Dedicated CPU
- ✅ Priority support

---

## 📝 Deployment Checklist

- [ ] GitHub repository created and code pushed
- [ ] Railway account created
- [ ] Project deployed on Railway
- [ ] Custom domain generated
- [ ] Environment variables set
- [ ] Backend URL tested (visit `/` endpoint)
- [ ] Frontend updated with new API URL
- [ ] CORS working (test from frontend)
- [ ] Video upload and analysis tested
- [ ] Logs monitored for errors

---

## 🎉 Success!

Your backend is now deployed! Test it:

1. Visit: `https://your-app.up.railway.app/`
2. Should see: API information with endpoints
3. Test health: `https://your-app.up.railway.app/health`
4. Update frontend and test video analysis

---

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Railway Discord](https://discord.gg/railway)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)

---

## 🆘 Need Help?

If you encounter issues:
1. Check Railway logs: `railway logs`
2. Verify environment variables
3. Check Railway status: [status.railway.app](https://status.railway.app)
4. Join Railway Discord for support

Good luck with your deployment! 🚀
