# 🚀 PPE Guard — Complete Hosting Guide
### By Sarthak Shirdhankar — Final Year Project

---

## PART 1 — FILE CHANGES SUMMARY

### Files you DO NOT touch / delete:
- `app.py` ← keep it as-is (still works for image upload mode)
- `train.py` ← keep it (original training script)
- `yolov8n.pt` ← keep it in your root folder for now

### Files you REPLACE (new versions provided):
| File | What changed |
|---|---|
| `Procfile` | Points to `app_realtime:app` instead of `app:app`, adds `--threads 4` |
| `requirements.txt` | Added `roboflow>=1.1.0` |
| `.gitignore` | Added `dataset/` and `*.onnx` / `*.engine` |

### Files you ADD (new, don't replace anything):
| File | Purpose |
|---|---|
| `realtime_pipeline.py` | Core real-time engine |
| `app_realtime.py` | New Flask backend with live stream |
| `train_optimized.py` | Better training script |
| `optimize.py` | Benchmarking and model export |

### Final folder structure after changes:
```
ppe_detector/
├── app.py                  ← KEEP (unchanged)
├── app_realtime.py         ← ADD (new)
├── realtime_pipeline.py    ← ADD (new)
├── train.py                ← KEEP (unchanged)
├── train_optimized.py      ← ADD (new)
├── optimize.py             ← ADD (new)
├── Procfile                ← REPLACE
├── requirements.txt        ← REPLACE
├── .gitignore              ← REPLACE
├── yolov8n.pt              ← KEEP (move to models/ before deploy)
├── models/
│   ├── .gitkeep
│   └── best.pt             ← your trained model goes here
├── uploads/
│   └── .gitkeep
└── templates/
    └── index.html          ← KEEP (unchanged)
```

---

## PART 2 — LOCAL SETUP (Run on your laptop first)

### Step 1: Create folder structure
Open terminal in your project folder and run:
```bash
mkdir -p models uploads static
touch models/.gitkeep uploads/.gitkeep
```

### Step 2: Move yolov8n.pt into models/
```bash
# Windows:
move yolov8n.pt models\yolov8n.pt

# Mac/Linux:
mv yolov8n.pt models/yolov8n.pt
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Test locally
```bash
# Test image upload mode (original):
python app.py

# Test real-time mode (new):
python app_realtime.py
```

Open browser → http://localhost:5000

For live camera, in a second terminal:
```bash
curl -X POST http://localhost:5000/api/start_camera \
  -H "Content-Type: application/json" \
  -d "{\"source\": 0}"
```

Then open: http://localhost:5000/stream in your browser.

---

## PART 3 — HOSTING ON RENDER.COM (Free, Recommended)

Render is the best free option for your project. No credit card needed.

### Step 1: Push to GitHub

1. Go to github.com → New Repository → name it `ppe-guard`
2. Make it **Public**
3. In your project terminal:

```bash
git init
git add .
git commit -m "Initial PPE Guard commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ppe-guard.git
git push -u origin main
```

> ⚠️ IMPORTANT: The `.gitignore` has `*.pt` so your model file will NOT be pushed.
> This is correct — model files are too large for GitHub (100MB limit).
> You will upload the model separately on Render (see Step 4).

### Step 2: Create Render Account
1. Go to render.com
2. Sign up with your GitHub account
3. Click "New +" → "Web Service"
4. Connect your GitHub → select `ppe-guard` repo

### Step 3: Configure Render Settings
Fill in these exact values:

```
Name:            ppe-guard
Region:          Singapore (closest to India)
Branch:          main
Runtime:         Python 3
Build Command:   pip install -r requirements.txt
Start Command:   gunicorn app_realtime:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4
```

Plan: **Free** (512MB RAM, enough for yolov8n)

### Step 4: Add Model File to Render

Render's free tier does NOT have persistent disk by default.
Two options:

**Option A — Use yolov8n.pt as fallback (easiest, works now):**
The app already falls back to `yolov8n.pt` automatically.
Remove `*.pt` from `.gitignore` ONLY for the base model:

In `.gitignore`, change:
```
*.pt
```
to:
```
*.pt
!models/yolov8n.pt
```

Then:
```bash
cp yolov8n.pt models/yolov8n.pt   # or move it
git add models/yolov8n.pt
git commit -m "Add base model"
git push
```

> Note: yolov8n.pt is 6.3MB — well within GitHub's limits.

**Option B — Upload trained best.pt to cloud storage:**
1. Upload `models/best.pt` to Google Drive or Hugging Face
2. Add a download step in your `app_realtime.py` startup:

```python
# Add this to app_realtime.py before load_model():
import urllib.request
MODEL_URL = "https://your-gdrive-direct-link/best.pt"
if not os.path.exists("models/best.pt"):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, "models/best.pt")
```

### Step 5: Environment Variables on Render
In Render → your service → "Environment":

```
PYTHON_VERSION = 3.10.0
```

If you want Roboflow dataset download:
```
ROBOFLOW_API_KEY = your_key_here
```

### Step 6: Deploy
Click "Create Web Service" → Render builds and deploys automatically.
First deploy takes ~5 minutes.

Your app will be live at: `https://ppe-guard.onrender.com`

> ⚠️ Free Render tier spins down after 15 minutes of inactivity.
> First request after spin-down takes ~30 seconds to wake up.
> This is normal for the free plan.

---

## PART 4 — HOSTING ON RAILWAY.APP (Alternative, also free)

Railway gives you $5/month free credits — enough for ~500 hours.

### Step 1: Go to railway.app → New Project → Deploy from GitHub
### Step 2: Select your repo
### Step 3: Add these environment variables:
```
PORT = 5000
PYTHON_VERSION = 3.10
```
### Step 4: Railway auto-detects Procfile and deploys.

Your URL: `https://ppe-guard-production.up.railway.app`

---

## PART 5 — HOSTING ON A VPS / CLOUD VM (For Real-Time Camera)

> Use this if you need REAL live camera streams (Render/Railway don't support USB cameras).
> Best for your actual construction site deployment.

### Recommended: Oracle Cloud Free Tier (Always Free VM)
- Go to cloud.oracle.com → Free Tier
- Create an **AMD VM** (2 CPU, 1GB RAM — always free)
- OS: Ubuntu 22.04

### SSH into your VM:
```bash
ssh ubuntu@YOUR_VM_IP
```

### Install dependencies:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git libgl1
```

### Clone your project:
```bash
git clone https://github.com/YOUR_USERNAME/ppe-guard.git
cd ppe-guard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Copy your model to the VM:
```bash
# From your local machine:
scp models/best.pt ubuntu@YOUR_VM_IP:~/ppe-guard/models/
```

### Run with Gunicorn (production):
```bash
gunicorn app_realtime:app \
  --bind 0.0.0.0:5000 \
  --workers 1 \
  --threads 4 \
  --timeout 120 \
  --daemon   # runs in background
```

### Keep it running after logout (systemd service):
```bash
sudo nano /etc/systemd/system/ppeguard.service
```

Paste this:
```ini
[Unit]
Description=PPE Guard Detection Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ppe-guard
Environment="PATH=/home/ubuntu/ppe-guard/venv/bin"
ExecStart=/home/ubuntu/ppe-guard/venv/bin/gunicorn app_realtime:app --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 120
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ppeguard
sudo systemctl start ppeguard
sudo systemctl status ppeguard   # should say "active (running)"
```

Access at: `http://YOUR_VM_IP:5000`

---

## PART 6 — CONNECTING A REAL IP CAMERA

### USB Camera (local machine only):
```bash
python realtime_pipeline.py --source 0   # first USB cam
python realtime_pipeline.py --source 1   # second USB cam
```

### IP Camera / CCTV (RTSP stream):
Most IP cameras have RTSP. Format:
```
rtsp://username:password@192.168.1.64:554/stream
rtsp://admin:admin123@192.168.1.100/h264/ch1/main/av_stream
```

Start via API:
```bash
curl -X POST http://localhost:5000/api/start_camera \
  -H "Content-Type: application/json" \
  -d "{\"source\": \"rtsp://admin:admin@192.168.1.64/stream\"}"
```

### Find your camera's RTSP URL:
- Check your camera's manual or admin panel
- Try: `rtsp://IP_ADDRESS/stream` or `rtsp://IP_ADDRESS:554/live`
- Use VLC → Media → Open Network Stream to test the URL first

---

## PART 7 — COMMON ERRORS AND FIXES

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: realtime_pipeline` | New files not added | Add all 4 new `.py` files to your folder |
| `No module named cv2` | OpenCV missing | `pip install opencv-python-headless` |
| `OSError: [Errno 28] No space left` | Model too large for Render free | Use yolov8n.pt (6MB) not best.pt (100MB+) |
| `gunicorn: command not found` | gunicorn not installed | `pip install gunicorn` |
| Camera won't start on Render | Render has no camera | Use Render only for image upload mode; use VPS for live cam |
| `timeout` errors on Render | Free plan is slow | Upgrade to $7/month Starter, or use Railway |
| Stream not loading | MJPEG blocked | Make sure you're using `http://`, not HTTPS, for the stream URL |

---

## PART 8 — CHECKLIST BEFORE GOING LIVE

- [ ] All 4 new files added to project folder
- [ ] `Procfile` updated (points to `app_realtime:app`)
- [ ] `requirements.txt` updated
- [ ] `models/yolov8n.pt` committed to GitHub (or best.pt uploaded separately)
- [ ] `models/.gitkeep` and `uploads/.gitkeep` exist
- [ ] Tested locally with `python app_realtime.py`
- [ ] GitHub repo is public
- [ ] Render/Railway connected to GitHub repo
- [ ] First deploy successful (check Render logs for errors)
- [ ] `/api/status` returns `{"model_loaded": true}`

---

*Good luck with your final year project, Sarthak! 🦺*
