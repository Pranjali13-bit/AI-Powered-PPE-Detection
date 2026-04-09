# 🦺 PPE Guard — AI Construction Safety Detection

Final Year Engineering Project by **Sarthak Shirdhankar**

---

## ⚡ Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open browser → http://localhost:5000
```

The app runs in **Demo Mode** without a trained model. To use real AI detection:

---

## 🤖 Training Your Own Model

### Step 1: Get a PPE Dataset (Free)
- Go to [Roboflow Universe](https://universe.roboflow.com/)
- Search: **"Construction Site Safety"**
- Download in **YOLOv8 format** → extract to `./dataset/`

### Step 2: Train
```bash
python train.py
```
Training takes ~30 min on CPU, ~5 min on GPU. Creates `models/best.pt`.

### Step 3: Run with Real Model
```bash
python app.py
```

---

## 🌐 Deploy to Render (Free Hosting)

1. Push folder to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Deploy ✅

---

## 📁 Project Structure

```
ppe_detector/
├── app.py              ← Flask backend + detection logic
├── train.py            ← YOLOv8 training script
├── requirements.txt    ← Python dependencies
├── Procfile            ← For Render/Heroku deployment
├── models/
│   └── best.pt         ← Your trained model (place here)
├── templates/
│   └── index.html      ← Frontend UI
├── static/             ← CSS/JS/images
└── uploads/            ← Temporary file storage
```

---

## 🎯 Detected PPE Classes

| Icon | Equipment | Category |
|------|-----------|----------|
| 🪖 | Safety Helmet | Head Protection |
| 🥽 | Safety Glasses/Goggles | Eye Protection |
| 🎧 | Ear Plugs/Muffs | Hearing Protection |
| 🧤 | Safety Gloves | Hand Protection |
| 👟 | Safety Shoes/Boots | Foot Protection |
| 🦺 | Safety Vest | Body Protection |

---

## 👨‍💻 Developer

**Sarthak Shirdhankar**  
Final Year B.Tech — Construction Safety AI  
📞 +91 96078 66693  
✉️ shirdhankarsarthak@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sarthak-shirdhankar-3b69a9386)
