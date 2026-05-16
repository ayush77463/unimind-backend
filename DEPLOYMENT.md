# UniMind Backend Deployment Guide

Deploy the UniMind memory backend to **Render** (free tier) so the Flutter APK
works from any phone on any network.

---

## Prerequisites

1. **GitHub account** — [github.com](https://github.com)
2. **Render account** — [render.com](https://render.com) (sign up with GitHub)
3. **Gemini API key** — [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## Step 1: Push Backend to GitHub

Create a **new GitHub repository** for the backend only.

```bash
# From your project root, navigate to the backend folder
cd unimind_memory

# Initialize a git repo (if not already)
git init

# Add all backend files
git add .

# Commit
git commit -m "Initial UniMind backend for deployment"

# Create a new repo on GitHub (e.g., unimind-backend), then:
git remote add origin https://github.com/YOUR_USERNAME/unimind-backend.git
git branch -M main
git push -u origin main
```

> **Important**: The `.gitignore` will prevent `.env` (with your API key) and
> `storage/` from being committed. Never push secrets to GitHub.

---

## Step 2: Create Render Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account (if not already connected)
4. Select your **`unimind-backend`** repository
5. Configure:

| Setting | Value |
|---|---|
| **Name** | `unimind-api` |
| **Region** | Oregon (US West) or closest to you |
| **Branch** | `main` |
| **Runtime** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

6. Click **"Create Web Service"**

---

## Step 3: Add Persistent Disk

This keeps your SQLite database and FAISS index across deploys.

1. In your Render service dashboard → **"Disks"** tab
2. Click **"Add Disk"**
3. Configure:

| Setting | Value |
|---|---|
| **Name** | `unimind-storage` |
| **Mount Path** | `/opt/render/project/src/storage` |
| **Size** | 1 GB (free tier) |

4. Save

---

## Step 4: Set Environment Variables

In your Render service dashboard → **"Environment"** tab:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from Google AI Studio |
| `RENDER_DISK_PATH` | `/opt/render/project/src/storage` |
| `PYTHON_VERSION` | `3.11.9` |

Click **"Save Changes"**. The service will auto-redeploy.

---

## Step 5: Verify Deployment

After Render finishes building (2-5 minutes), test these URLs in your browser:

- **Health check**: `https://unimind-api.onrender.com/health`
- **API docs**: `https://unimind-api.onrender.com/docs`
- **Root**: `https://unimind-api.onrender.com/`

You should see JSON responses confirming the API is running.

> **Note**: First request may take ~30 seconds if the service was sleeping
> (free tier spins down after 15 minutes of inactivity).

---

## Step 6: Update Flutter APK

The Flutter app is already updated to use `https://unimind-api.onrender.com`
as the default backend URL. Just rebuild the APK:

```bash
cd c:\Codes\Project 4\unimind_v7
flutter build apk --release
```

Or build a debug APK for testing:

```bash
flutter build apk --debug
```

The app will automatically connect to the cloud backend. Users can also
override the URL in **Preferences → Backend Server** if needed.

---

## How to Redeploy Updates

When you make changes to the backend code:

```bash
cd unimind_memory
git add .
git commit -m "Update backend"
git push origin main
```

Render will **automatically redeploy** within 2-3 minutes.

---

## How to Monitor

1. **Render Dashboard** → Your service → **"Logs"** tab
2. Shows real-time server logs, errors, and request activity
3. **"Events"** tab shows deploy history

---

## How to Restart

If the backend becomes unresponsive:

1. Go to Render Dashboard → Your service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**
3. Or click the **three dots menu** → **"Restart"**

---

## Changing API Keys

### From the App
The Gemini API key used for **direct chat** (Gemini, GPT, etc.) is managed
in the Flutter app under **Settings → Model Connectivity**. You can change
these anytime without touching the backend.

### Backend Gemini Key
The backend's Gemini API key (used for memory extraction and embeddings)
is set in Render's **Environment Variables**. To change it:

1. Render Dashboard → Your service → **"Environment"**
2. Update `GEMINI_API_KEY`
3. Save → auto-redeploy

---

## Troubleshooting

### "502 Bad Gateway" or "Service Unavailable"
- The service is still building or restarting. Wait 2-3 minutes.
- Check the **Logs** tab for errors.

### "Connection timed out" from the app
- Free tier services sleep after 15 min inactivity. First request wakes it up (~30s).
- Check your internet connection on the phone.

### Memory not persisting across deploys
- Make sure the **Persistent Disk** is configured (Step 3).
- Verify `RENDER_DISK_PATH` environment variable is set.

### CORS errors
- The backend already allows all origins (`allow_origins=["*"]`).
- If issues persist, check the Render logs for the actual error.

### App shows "Local" instead of "Live"
- The cloud URL may be sleeping. Open `https://unimind-api.onrender.com/health`
  in a browser first to wake it up.
- Check **Preferences → Backend Server** — ensure the URL is correct or empty
  (empty uses the default cloud URL).

---

## Architecture Summary

```
Flutter APK (any phone, any network)
    │
    ├── Gemini/GPT/Claude APIs (direct, using app API keys)
    │
    └── UniMind Memory Backend (Render cloud)
            │
            ├── FastAPI server
            ├── SQLite database (persistent disk)
            ├── FAISS vector index (persistent disk)
            ├── Memory extraction (Gemini API)
            └── Semantic retrieval
```

All AI chat happens directly from the app to the AI providers.
The backend handles only **memory storage, retrieval, and context building**.
