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
from collections import deque

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
USE_MOCK = False

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
        print("[MODEL] Downloading yolov8n.pt...")
        model = YOLO("yolov8n.pt")
        os.makedirs("models", exist_ok=True)
        model.save("models/yolov8n.pt")
        print("[MODEL] Ready")
    except Exception as e:
        print(f"[MODEL] Failed: {e}")
        USE_MOCK = True

# ── Step 1: Detect PERSONS with YOLO (COCO class 0) ───────────────────────────
def detect_persons(image_np):
    if model is None or USE_MOCK:
        return mock_persons(image_np)
    try:
        results = model(image_np, conf=0.20, iou=0.40, classes=[0], verbose=False)
        persons = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                if (y2-y1) < 40 or (x2-x1) < 20:
                    continue
                persons.append({
                    "bbox": [x1,y1,x2,y2],
                    "confidence": round(float(box.conf[0])*100,1)
                })
        return persons
    except Exception as e:
        print(f"[PERSONS] {e}")
        return mock_persons(image_np)

def mock_persons(image_np):
    h, w = image_np.shape[:2]
    return [
        {"bbox":[int(w*.03),int(h*.05),int(w*.28),int(h*.95)],"confidence":91.2},
        {"bbox":[int(w*.30),int(h*.03),int(w*.60),int(h*.95)],"confidence":88.7},
        {"bbox":[int(w*.62),int(h*.08),int(w*.95),int(h*.92)],"confidence":84.1},
    ]

# ── Step 2: Colour-based PPE analysis per person ───────────────────────────────
def analyze_ppe(image_np, bbox):
    """
    Analyses cropped person for PPE using HSV colour segmentation.
    - HELMET : hard-hat colours in top 28% of person box
    - VEST   : hi-vis colours in mid-body (22-72%)
    - GOGGLES: dark lens region at eye level (12-28%)
    Works with ANY model — no PPE training required.
    """
    x1,y1,x2,y2 = bbox
    h_img,w_img = image_np.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(w_img,x2); y2=min(h_img,y2)
    crop = image_np[y1:y2, x1:x2]
    if crop.size == 0:
        return False,False,False,0,0,0

    ch,cw = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # ── HELMET (top 28%) ──────────────────────────────────────────────────────
    head = hsv[:int(ch*0.28), :]
    hpx  = max(head.shape[0]*head.shape[1], 1)
    masks_h = [
        cv2.inRange(head,(18,80,80),(38,255,255)),    # yellow
        cv2.inRange(head,(0,0,170),(180,45,255)),     # white
        cv2.inRange(head,(5,100,80),(18,255,255)),    # orange
        cv2.inRange(head,(0,80,60),(12,255,255)),     # red
        cv2.inRange(head,(165,80,60),(180,255,255)),  # red-wrap
        cv2.inRange(head,(95,60,60),(130,255,255)),   # blue
    ]
    hm = masks_h[0]
    for m in masks_h[1:]: hm = cv2.bitwise_or(hm,m)
    hr = cv2.countNonZero(hm)/hpx
    has_helmet  = hr > 0.06
    helmet_conf = min(99, int(hr*350))

    # ── VEST (22-72%) ─────────────────────────────────────────────────────────
    vest_z = hsv[int(ch*0.22):int(ch*0.72), :]
    vpx    = max(vest_z.shape[0]*vest_z.shape[1], 1)
    masks_v = [
        cv2.inRange(vest_z,(28,100,80),(88,255,255)),  # lime/yellow-green
        cv2.inRange(vest_z,(5,110,80),(22,255,255)),   # orange
    ]
    vm = cv2.bitwise_or(masks_v[0], masks_v[1])
    vr = cv2.countNonZero(vm)/vpx
    has_vest  = vr > 0.08
    vest_conf = min(99, int(vr*280))

    # ── GOGGLES (eye zone 12-28%, inner 70% width) ───────────────────────────
    ey = hsv[int(ch*0.12):int(ch*0.28), int(cw*0.15):int(cw*0.85)]
    epx = max(ey.shape[0]*ey.shape[1], 1)
    dark = cv2.inRange(ey,(0,0,0),(180,255,80))
    dr   = cv2.countNonZero(dark)/epx
    has_glasses  = dr > 0.25
    glasses_conf = min(99, int(dr*200))

    return has_helmet, has_vest, has_glasses, helmet_conf, vest_conf, glasses_conf

# ── Step 3: Build workers ──────────────────────────────────────────────────────
def build_workers(image_np, persons):
    workers = []
    for i,p in enumerate(persons):
        hh,hv,hg,hc,vc,gc = analyze_ppe(image_np, p["bbox"])
        compliant = hh and hv
        workers.append({
            "id":i+1, "bbox":p["bbox"], "person_conf":p["confidence"],
            "has_helmet":hh, "has_vest":hv, "has_glasses":hg,
            "helmet_conf":hc, "vest_conf":vc, "glasses_conf":gc,
            "compliant":compliant,
            "ppe_found":(["Safety Helmet"] if hh else [])+(["Safety Vest"] if hv else [])+(["Goggles"] if hg else []),
        })
    return workers

# ── Annotate ───────────────────────────────────────────────────────────────────
def annotate_image(image_np, workers):
    out = image_np.copy()
    GREEN=(0,200,80); RED=(0,60,220)

    for w in workers:
        x1,y1,x2,y2 = w["bbox"]
        col = GREEN if w["compliant"] else RED
        cv2.rectangle(out,(x1,y1),(x2,y2),col,3)

        label = f"W{w['id']} {'SAFE' if w['compliant'] else 'VIOLATION'}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.58,2)
        cv2.rectangle(out,(x1,y1-th-12),(x1+tw+8,y1),col,-1)
        cv2.putText(out,label,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2,cv2.LINE_AA)

        dot_y = y1+18
        for j,(lbl,found) in enumerate([("H",w["has_helmet"]),("V",w["has_vest"]),("G",w["has_glasses"])]):
            dc=GREEN if found else RED
            cx=x1+14+j*30
            cv2.circle(out,(cx,dot_y),11,dc,-1)
            cv2.putText(out,lbl,(cx-5,dot_y+5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)

        ch=y2-y1
        if w["has_helmet"]:
            cv2.rectangle(out,(x1,y1),(x2,y1+int(ch*0.28)),(0,215,255),1)
        if w["has_vest"]:
            cv2.rectangle(out,(x1,y1+int(ch*0.22)),(x2,y1+int(ch*0.72)),(200,0,255),1)

    # HUD
    total=len(workers); compliant=sum(1 for w in workers if w["compliant"])
    helmets=sum(1 for w in workers if w["has_helmet"])
    vests=sum(1 for w in workers if w["has_vest"])
    goggles=sum(1 for w in workers if w["has_glasses"])
    rate=int(compliant/total*100) if total else 0

    ov=out.copy(); cv2.rectangle(ov,(8,8),(245,195),(15,15,15),-1)
    cv2.addWeighted(ov,0.78,out,0.22,0,out)

    def hud(t,v,g,y):
        c=(0,200,80) if g else (0,60,220)
        cv2.putText(out,t,(16,y),cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,200),1,cv2.LINE_AA)
        cv2.putText(out,str(v),(190,y),cv2.FONT_HERSHEY_SIMPLEX,0.52,c,1,cv2.LINE_AA)

    cv2.putText(out,"PPE GUARD",(16,30),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,220,120),2,cv2.LINE_AA)
    hud("Workers:",total,True,55)
    hud("Helmets:",f"{helmets}/{total}",helmets==total,78)
    hud("Vests:",f"{vests}/{total}",vests==total,101)
    hud("Goggles:",f"{goggles}/{total}",goggles==total,124)
    hud("Compliant:",f"{compliant}/{total}",compliant==total,147)
    hud("Rate:",f"{rate}%",rate==100,170)
    if total and compliant<total:
        cv2.putText(out,"!! VIOLATION !!",(16,193),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,60,220),2,cv2.LINE_AA)
    return out

def img2b64(img):
    _,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,90])
    return base64.b64encode(buf).decode()

def load_image_bytes(fb, filename=""):
    try:
        nparr=np.frombuffer(fb,np.uint8)
        img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        if img is not None: return img
        pil=Image.open(io.BytesIO(fb)).convert("RGB")
        return cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
    except: return None

# ── Camera ─────────────────────────────────────────────────────────────────────
camera_state={"running":False,"cap":None,"latest_annotated":None,
              "latest_counts":{},"fps":0,"thread":None,"lock":threading.Lock()}

def camera_loop():
    cs=camera_state; fps_times=deque(maxlen=20)
    while cs["running"]:
        cap=cs.get("cap")
        if not cap or not cap.isOpened(): time.sleep(0.05); continue
        ret,frame=cap.read()
        if not ret: time.sleep(0.02); continue
        t0=time.perf_counter()
        persons=detect_persons(frame)
        workers=build_workers(frame,persons)
        ann=annotate_image(frame,workers)
        fps_times.append(time.perf_counter()-t0)
        fps=len(fps_times)/sum(fps_times) if fps_times else 0
        total=len(workers); compliant=sum(1 for w in workers if w["compliant"])
        with cs["lock"]:
            cs["latest_annotated"]=ann; cs["fps"]=round(fps,1)
            cs["latest_counts"]={
                "fps":round(fps,1),"total":total,"compliant":compliant,
                "violations":total-compliant,
                "helmets":sum(1 for w in workers if w["has_helmet"]),
                "vests":sum(1 for w in workers if w["has_vest"]),
                "goggles":sum(1 for w in workers if w["has_glasses"]),
                "rate":int(compliant/total*100) if total else 0,
                "workers":[{"id":w["id"],"compliant":w["compliant"],
                             "has_helmet":w["has_helmet"],"has_vest":w["has_vest"],
                             "has_glasses":w["has_glasses"]} for w in workers]
            }

def gen_stream():
    while True:
        with camera_state["lock"]:
            frame=camera_state.get("latest_annotated")
        if frame is None:
            ph=np.zeros((480,640,3),dtype=np.uint8)
            cv2.putText(ph,"Waiting for camera...",(140,240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,100,100),2)
            frame=ph
        _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,72])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        time.sleep(0.04)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html",demo_mode=USE_MOCK)

@app.route("/stream")
def stream(): return Response(gen_stream(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/start",methods=["POST"])
def camera_start():
    cs=camera_state
    if cs["running"]: return jsonify({"status":"already_running"})
    data=request.json or {}; source=data.get("source",0)
    if isinstance(source,str) and source.isdigit(): source=int(source)
    cap=cv2.VideoCapture(source); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened(): return jsonify({"error":"Cannot open camera"}),400
    cs["cap"]=cap; cs["running"]=True
    t=threading.Thread(target=camera_loop,daemon=True); cs["thread"]=t; t.start()
    return jsonify({"status":"started"})

@app.route("/api/camera/stop",methods=["POST"])
def camera_stop():
    cs=camera_state; cs["running"]=False
    if cs.get("cap"): cs["cap"].release(); cs["cap"]=None
    with cs["lock"]: cs["latest_annotated"]=None; cs["latest_counts"]={}
    return jsonify({"status":"stopped"})

@app.route("/api/camera/counts")
def camera_counts():
    with camera_state["lock"]: return jsonify({"success":True,"data":camera_state["latest_counts"]})

@app.route("/api/detect",methods=["POST"])
def detect():
    try:
        if request.is_json:
            data=request.json; img_data=data.get("image","")
            if "," in img_data: img_data=img_data.split(",")[1]
            image_np=load_image_bytes(base64.b64decode(img_data),".jpg")
        else:
            f=request.files.get("file")
            if not f: return jsonify({"error":"No file"}),400
            image_np=load_image_bytes(f.read(),f.filename)
        if image_np is None: return jsonify({"error":"Cannot read image"}),400

        persons=detect_persons(image_np)
        workers=build_workers(image_np,persons)
        ann=annotate_image(image_np,workers)
        total=len(workers); compliant=sum(1 for w in workers if w["compliant"])

        detections=[]
        for w in workers:
            detections.append({"class_id":7,"class_name":"Person","icon":"🧍","color":"#B0BEC5","confidence":w["person_conf"],"bbox":w["bbox"]})
            if w["has_helmet"]: detections.append({"class_id":0,"class_name":"Safety Helmet","icon":"🪖","color":"#FFD700","confidence":w["helmet_conf"],"bbox":w["bbox"]})
            if w["has_vest"]:   detections.append({"class_id":5,"class_name":"Safety Vest","icon":"🦺","color":"#FF4FCE","confidence":w["vest_conf"],"bbox":w["bbox"]})
            if w["has_glasses"]:detections.append({"class_id":1,"class_name":"Safety Glasses","icon":"🥽","color":"#00FFFF","confidence":w["glasses_conf"],"bbox":w["bbox"]})

        return jsonify({
            "success":True,"annotated_image":f"data:image/jpeg;base64,{img2b64(ann)}",
            "detections":detections,"workers":workers,"count":len(detections),
            "summary":{"total_workers":total,"compliant":compliant,"violations":total-compliant,
                       "helmets":sum(1 for w in workers if w["has_helmet"]),
                       "vests":sum(1 for w in workers if w["has_vest"]),
                       "goggles":sum(1 for w in workers if w["has_glasses"]),
                       "compliance_rate":int(compliant/total*100) if total else 0},
            "demo_mode":USE_MOCK,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/api/detect_video_frame",methods=["POST"])
def detect_video_frame():
    try:
        data=request.json; img_data=data.get("frame","")
        if "," in img_data: img_data=img_data.split(",")[1]
        nparr=np.frombuffer(base64.b64decode(img_data),np.uint8)
        image_np=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        if image_np is None: return jsonify({"error":"Invalid frame"}),400
        persons=detect_persons(image_np); workers=build_workers(image_np,persons)
        ann=annotate_image(image_np,workers)
        total=len(workers); compliant=sum(1 for w in workers if w["compliant"])
        return jsonify({"success":True,"annotated_frame":f"data:image/jpeg;base64,{img2b64(ann)}","count":len(workers),
                        "summary":{"total_workers":total,"compliant":compliant,"violations":total-compliant,
                                   "helmets":sum(1 for w in workers if w["has_helmet"]),
                                   "vests":sum(1 for w in workers if w["has_vest"]),
                                   "goggles":sum(1 for w in workers if w["has_glasses"]),
                                   "compliance_rate":int(compliant/total*100) if total else 0}})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/status")
def status():
    return jsonify({"model_loaded":model is not None,"demo_mode":USE_MOCK,
                    "camera_running":camera_state["running"],
                    "ppe_classes":[{"id":k,**v} for k,v in PPE_CLASSES.items()]})

if __name__=="__main__":
    os.makedirs("uploads",exist_ok=True); os.makedirs("models",exist_ok=True)
    load_model()
    print("\n🦺 PPE Guard → http://localhost:5000\n")
    app.run(debug=False,host="0.0.0.0",port=5000)
