from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Class IDs (CHANGE if your model uses different labels)
PERSON_ID = 0
HELMET_ID = 1
VEST_ID = 2
GOGGLES_ID = 3

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]

        detections = []
        ppe_boxes = []

        # Collect detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == PERSON_ID:
                detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
            else:
                ppe_boxes.append((cls, x1, y1, x2, y2))

        # Track persons
        tracks = tracker.update_tracks(detections, frame=frame)

        persons = {}

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            px1, py1, px2, py2 = l, t, l+w, t+h

            persons[track_id] = {
                "box": (px1, py1, px2, py2),
                "helmet": False,
                "vest": False,
                "goggles": False
            }

        # Assign PPE to person
        for cls, x1, y1, x2, y2 in ppe_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            for pid, pdata in persons.items():
                px1, py1, px2, py2 = pdata["box"]

                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    if cls == HELMET_ID:
                        pdata["helmet"] = True
                    elif cls == VEST_ID:
                        pdata["vest"] = True
                    elif cls == GOGGLES_ID:
                        pdata["goggles"] = True

        # Count results
        total_persons = len(persons)
        helmet_count = sum(p["helmet"] for p in persons.values())
        vest_count = sum(p["vest"] for p in persons.values())
        goggles_count = sum(p["goggles"] for p in persons.values())

        # Draw boxes
        for pid, pdata in persons.items():
            x1, y1, x2, y2 = pdata["box"]

            label = f"ID {pid}"

            if all([pdata["helmet"], pdata["vest"], pdata["goggles"]]):
                color = (0, 255, 0)
            elif any([pdata["helmet"], pdata["vest"], pdata["goggles"]]):
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display counts
        cv2.putText(frame, f"Persons: {total_persons}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Helmet: {helmet_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Vest: {vest_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Goggles: {goggles_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <h2>PPE Detection (Live)</h2>
    <img src="/video_feed">
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)