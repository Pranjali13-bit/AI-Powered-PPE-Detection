import os
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import io
import time
import threading
import queue
from collections import deque

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
USE_MOCK = False

# ── Class Definitions ──────────────────────────────────────────────────────────
PPE_CLASSES = {
    0: {"name": "Safety Helmet", "icon": "🪖", "color": "#FFD700", "key": "helmet"},
    1: {"name": "Safety Glasses","icon": "🥽", "color": "#00FFFF", "key": "glasses"},
    2: {"name": "Ear Protection","icon": "🎧", "color": "#FF6B6B", "key": "ear"},
    3: {"name": "Safety Gloves", "icon": "🧤", "color": "#A8FF78", "key": "gloves"},
    4: {"name": "Safety Shoes",  "icon": "👟", "color": "#FF9500", "key": "shoes"},
    5: {"name": "Safety Vest",   "icon": "🦺", "color": "#FF4FCE", "key": "vest"},
    6: {"name": "Face Shield",   "icon": "🛡️", "color": "#4FC3F7", "key": "shield"},
    7: {"name": "Person",        "icon": "🧍", "color": "#B0BEC5", "key": "person"},
}

ROBOFLOW_CLASSES = {
    "hardhat": 0, "helmet": 0, "hard hat": 0, "safety helmet": 0,
    "glasses": 1, "goggles": 1, "safety glasses": 1,
    "ear": 2, "ear protection": 2,
    "gloves": 3, "safety gloves": 3, "hand": 3,
    "boots": 4, "shoes": 4, "safety shoes": 4, "footwear": 4,
    "vest": 5, "safety vest": 5,
    "mask": 6, "face shield": 6, "shield": 6,
    "person": 7, "worker": 7,
}

# ── Model Loading ──────────────────────────────────────────────────────────────
def load_model():
    global model, USE_MOCK
    paths = ["models/best.pt", "models/yolov8n.pt", "yolov8n.pt"]
    try:
        from ultralytics import YOLO
        for p in paths:
            if os.path.exists(p):
                model = YOLO(p)
                print(f"[MODEL] Loaded: {p}")
                return
        model = YOLO("yolov8n.pt")
        os.makedirs("models", exist_ok=True)
        model.save("models/yolov8n.pt")
    except Exception as e:
        print(f"[MODEL] Failed: {e}")
        USE_MOCK = True

# ── Class Name Mapper ──────────────────────────────────────────────────────────
def map_class(name, cid):
    ln = name.lower()
    for k, v in ROBOFLOW_CLASSES.items():
        if k in ln or ln in k:
            return v
    if cid == 0 and name == "person":
        return 7
    return cid % len(PPE_CLASSES)

# ── Core: Person-PPE Association ───────────────────────────────────────────────
def associate_ppe_to_persons(raw_detections, expand=0.25, max_dist=200):
    """
    Takes flat list of detections, returns list of Worker dicts:
    {
        track_id, bbox,
        has_helmet, has_vest, has_glasses,
        compliant, ppe_found: [...],
        confidence: {...}
    }
    Also returns ppe_items list for drawing.
    """
    persons  = [d for d in raw_detections if d["class_id"] == 7]
    ppe_items = [d for d in raw_detections if d["class_id"] != 7]

    workers = []
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p["bbox"]
        w, h = x2 - x1, y2 - y1
        workers.append({
            "id": i + 1,
            "bbox": p["bbox"],
            "person_conf": p["confidence"],
            "has_helmet":  False,
            "has_vest":    False,
            "has_glasses": False,
            "ppe_found":   [],
            "confidence":  {},
            "compliant":   False,
        })

    for ppe in ppe_items:
        px = (ppe["bbox"][0] + ppe["bbox"][2]) // 2
        py = (ppe["bbox"][1] + ppe["bbox"][3]) // 2

        best_wid, best_score = None, float("inf")
        for w in workers:
            bx1, by1, bx2, by2 = w["bbox"]
            bw = bx2 - bx1; bh = by2 - by1
            ex1 = bx1 - int(bw * expand)
            ey1 = by1 - int(bh * expand)
            ex2 = bx2 + int(bw * expand)
            ey2 = by2 + int(bh * expand)
            inside = ex1 <= px <= ex2 and ey1 <= py <= ey2
            cx = (bx1 + bx2) // 2; cy = (by1 + by2) // 2
            dist = ((px - cx)**2 + (py - cy)**2) ** 0.5
            if inside and dist < best_score:
                best_score = dist
                best_wid = w["id"]
            elif not inside and dist < max_dist and dist < best_score:
                best_score = dist
                best_wid = w["id"]

        if best_wid is not None:
            target = next(w for w in workers if w["id"] == best_wid)
            key = PPE_CLASSES.get(ppe["class_id"], {}).get("key", "")
            target["ppe_found"].append(ppe["class_name"])
            target["confidence"][key] = ppe["confidence"]
            if key == "helmet":
                target["has_helmet"] = True
            elif key == "vest":
                target["has_vest"] = True
            elif key in ("glasses", "shield"):
                target["has_glasses"] = True

    for w in workers:
        w["compliant"] = w["has_helmet"] and w["has_vest"]

    return workers, ppe_items

# ── Detection ──────────────────────────────────────────────────────────────────
def run_detection(image_np):
    if USE_MOCK or model is None:
        return mock_detection(image_np)
    try:
        results = model(image_np, conf=0.25, iou=0.45, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                cid   = int(box.cls[0])
                cname = r.names.get(cid, f"class_{cid}")
                mid   = map_class(cname, cid)
                info  = PPE_CLASSES.get(mid, PPE_CLASSES[7])
                conf  = float(box.conf[0])
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                dets.append({
                    "class_id":   mid,
                    "class_name": info["name"],
                    "icon":       info["icon"],
                    "color":      info["color"],
                    "confidence": round(conf * 100, 1),
                    "bbox":       [x1, y1, x2, y2],
                    "original_class": cname,
                })
        return dets
    except Exception as e:
        print(f"[DETECT] {e}")
        return mock_detection(image_np)

def mock_detection(image_np):
    h, w = image_np.shape[:2]
    return [
        {"class_id":7,"class_name":"Person","icon":"🧍","color":"#B0BEC5","confidence":91.7,"bbox":[int(w*.05),int(h*.1),int(w*.3),int(h*.95)],"original_class":"person"},
        {"class_id":7,"class_name":"Person","icon":"🧍","color":"#B0BEC5","confidence":88.2,"bbox":[int(w*.35),int(h*.08),int(w*.65),int(h*.95)],"original_class":"person"},
        {"class_id":7,"class_name":"Person","icon":"🧍","color":"#B0BEC5","confidence":82.1,"bbox":[int(w*.68),int(h*.1),int(w*.95),int(h*.9)],"original_class":"person"},
        {"class_id":0,"class_name":"Safety Helmet","icon":"🪖","color":"#FFD700","confidence":87.3,"bbox":[int(w*.08),int(h*.08),int(w*.27),int(h*.22)],"original_class":"helmet"},
        {"class_id":5,"class_name":"Safety Vest","icon":"🦺","color":"#FF4FCE","confidence":83.1,"bbox":[int(w*.08),int(h*.25),int(w*.28),int(h*.6)],"original_class":"vest"},
        {"class_id":0,"class_name":"Safety Helmet","icon":"🪖","color":"#FFD700","confidence":79.4,"bbox":[int(w*.38),int(h*.06),int(w*.62),int(h*.2)],"original_class":"helmet"},
    ]

# ── Annotate Image ─────────────────────────────────────────────────────────────
def annotate_image(image_np, workers, ppe_items):
    out = image_np.copy()
    h_img, w_img = out.shape[:2]

    def hex2bgr(h):
        h = h.lstrip("#")
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return (b, g, r)

    # Draw PPE boxes (thin)
    for p in ppe_items:
        x1,y1,x2,y2 = p["bbox"]
        col = hex2bgr(p["color"])
        cv2.rectangle(out,(x1,y1),(x2,y2),col,1)

    # Draw person boxes with compliance color and badge
    for w in workers:
        x1,y1,x2,y2 = w["bbox"]
        col = (0,200,80) if w["compliant"] else (0,60,220)
        cv2.rectangle(out,(x1,y1),(x2,y2),col,3)

        # Header badge
        label = f"W{w['id']} {'✓ SAFE' if w['compliant'] else '✗ VIOLATION'}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out,(x1,y1-th-10),(x1+tw+8,y1),col,-1)
        cv2.putText(out,label,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)

        # PPE status dots inside box
        dot_y = y1 + 14
        items = [
            ("H", w["has_helmet"]),
            ("V", w["has_vest"]),
            ("G", w["has_glasses"]),
        ]
        for j,(lbl,found) in enumerate(items):
            dc = (0,200,80) if found else (0,60,220)
            cx = x1 + 12 + j*28
            cv2.circle(out,(cx,dot_y),10,dc,-1)
            cv2.putText(out,lbl,(cx-5,dot_y+5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)

    # ── HUD Panel ─────────────────────────────────────────────────────────────
    total     = len(workers)
    compliant = sum(1 for w in workers if w["compliant"])
    helmets   = sum(1 for w in workers if w["has_helmet"])
    vests     = sum(1 for w in workers if w["has_vest"])
    goggles   = sum(1 for w in workers if w["has_glasses"])
    rate      = int(compliant/total*100) if total else 0

    panel_w, panel_h = 230, 185
    overlay = out.copy()
    cv2.rectangle(overlay,(8,8),(8+panel_w,8+panel_h),(15,15,15),-1)
    cv2.addWeighted(overlay,0.78,out,0.22,0,out)

    def hud_line(text, val, good, y):
        col = (0,200,80) if good else (0,60,220)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(out, str(val), (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

    cv2.putText(out,"PPE GUARD",(16,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,120),2,cv2.LINE_AA)
    hud_line("Workers:",    total,                 True,            55)
    hud_line("Helmets:",    f"{helmets}/{total}",  helmets==total,  78)
    hud_line("Vests:",      f"{vests}/{total}",    vests==total,    101)
    hud_line("Goggles:",    f"{goggles}/{total}",  goggles==total,  124)
    hud_line("Compliant:",  f"{compliant}/{total}",compliant==total,147)
    hud_line("Rate:",       f"{rate}%",            rate==100,       170)

    if total > 0 and compliant < total:
        cv2.putText(out,"!! VIOLATION !!",(16,195),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,60,220),2,cv2.LINE_AA)

    return out

def img2b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()

def load_image_bytes(fb, filename=""):
    ext = Path(filename).suffix.lower()
    try:
        nparr = np.frombuffer(fb, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        pil = Image.open(io.BytesIO(fb)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        return None

# ── Camera State ───────────────────────────────────────────────────────────────
camera_state = {
    "running": False,
    "cap": None,
    "latest_frame": None,
    "latest_annotated": None,
    "latest_counts": {},
    "fps": 0,
    "thread": None,
    "lock": threading.Lock(),
}

def camera_loop():
    cs = camera_state
    fps_times = deque(maxlen=20)
    while cs["running"]:
        cap = cs.get("cap")
        if not cap or not cap.isOpened():
            time.sleep(0.05)
            continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        t0 = time.perf_counter()
        dets = run_detection(frame)
        workers, ppe_items = associate_ppe_to_persons(dets)
        ann = annotate_image(frame, workers, ppe_items)
        fps_times.append(time.perf_counter() - t0)
        fps = len(fps_times) / sum(fps_times) if fps_times else 0

        total     = len(workers)
        compliant = sum(1 for w in workers if w["compliant"])
        helmets   = sum(1 for w in workers if w["has_helmet"])
        vests     = sum(1 for w in workers if w["has_vest"])
        goggles   = sum(1 for w in workers if w["has_glasses"])

        with cs["lock"]:
            cs["latest_annotated"] = ann
            cs["fps"] = round(fps, 1)
            cs["latest_counts"] = {
                "fps": round(fps, 1),
                "total": total,
                "compliant": compliant,
                "violations": total - compliant,
                "helmets": helmets,
                "vests": vests,
                "goggles": goggles,
                "rate": int(compliant/total*100) if total else 0,
                "workers": [
                    {
                        "id": w["id"],
                        "compliant": w["compliant"],
                        "has_helmet": w["has_helmet"],
                        "has_vest": w["has_vest"],
                        "has_glasses": w["has_glasses"],
                        "ppe_found": w["ppe_found"],
                    }
                    for w in workers
                ]
            }

def gen_stream():
    while True:
        with camera_state["lock"]:
            frame = camera_state.get("latest_annotated")
        if frame is None:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph,"Waiting for camera...",(140,240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,100,100),2)
            frame = ph
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.04)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", demo_mode=USE_MOCK)

@app.route("/stream")
def stream():
    from flask import Response, stream_with_context
    return Response(stream_with_context(gen_stream()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    cs = camera_state
    if cs["running"]:
        return jsonify({"status": "already_running"})
    data   = request.json or {}
    source = data.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open camera"}), 400
    cs["cap"]     = cap
    cs["running"] = True
    t = threading.Thread(target=camera_loop, daemon=True)
    cs["thread"] = t
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    cs = camera_state
    cs["running"] = False
    if cs.get("cap"):
        cs["cap"].release()
        cs["cap"] = None
    with cs["lock"]:
        cs["latest_annotated"] = None
        cs["latest_counts"]    = {}
    return jsonify({"status": "stopped"})

@app.route("/api/camera/counts")
def camera_counts():
    with camera_state["lock"]:
        return jsonify({"success": True, "data": camera_state["latest_counts"]})

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if request.is_json:
            data = request.json
            img_data = data.get("image","")
            if "," in img_data: img_data = img_data.split(",")[1]
            fb = base64.b64decode(img_data)
            image_np = load_image_bytes(fb, ".jpg")
        else:
            f = request.files.get("file")
            if not f: return jsonify({"error":"No file"}),400
            image_np = load_image_bytes(f.read(), f.filename)

        if image_np is None:
            return jsonify({"error":"Cannot read image"}),400

        dets = run_detection(image_np)
        workers, ppe_items = associate_ppe_to_persons(dets)
        ann = annotate_image(image_np, workers, ppe_items)

        total     = len(workers)
        compliant = sum(1 for w in workers if w["compliant"])

        return jsonify({
            "success":          True,
            "annotated_image":  f"data:image/jpeg;base64,{img2b64(ann)}",
            "detections":       dets,
            "workers":          workers,
            "count":            len(dets),
            "summary": {
                "total_workers": total,
                "compliant":     compliant,
                "violations":    total - compliant,
                "helmets":       sum(1 for w in workers if w["has_helmet"]),
                "vests":         sum(1 for w in workers if w["has_vest"]),
                "goggles":       sum(1 for w in workers if w["has_glasses"]),
                "compliance_rate": int(compliant/total*100) if total else 0,
            },
            "demo_mode": USE_MOCK,
        })
    except Exception as e:
        print(f"[API] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect_video_frame", methods=["POST"])
def detect_video_frame():
    try:
        data = request.json
        img_data = data.get("frame","")
        if "," in img_data: img_data = img_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None: return jsonify({"error":"Invalid frame"}),400

        dets = run_detection(image_np)
        workers, ppe_items = associate_ppe_to_persons(dets)
        ann = annotate_image(image_np, workers, ppe_items)

        total     = len(workers)
        compliant = sum(1 for w in workers if w["compliant"])

        return jsonify({
            "success":        True,
            "detections":     dets,
            "workers":        workers,
            "annotated_frame":f"data:image/jpeg;base64,{img2b64(ann)}",
            "count":          len(dets),
            "summary": {
                "total_workers":   total,
                "compliant":       compliant,
                "violations":      total - compliant,
                "helmets":         sum(1 for w in workers if w["has_helmet"]),
                "vests":           sum(1 for w in workers if w["has_vest"]),
                "goggles":         sum(1 for w in workers if w["has_glasses"]),
                "compliance_rate": int(compliant/total*100) if total else 0,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "demo_mode":    USE_MOCK,
        "camera_running": camera_state["running"],
        "ppe_classes":  [{"id":k,**v} for k,v in PPE_CLASSES.items()],
    })

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    load_model()
    print("\n🦺 PPE Guard running → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
