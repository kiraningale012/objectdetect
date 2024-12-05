from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tempfile
import os
import threading
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLO model
model = YOLO('best.pt')
conversion_factor = 1.8889  # mm per pixel

# Temporary path for uploaded video
video_path = None
camera_running = False

@app.route('/')
def index():
    return render_template('index.html')




@socketio.on('start_camera_stream')
def start_camera_stream():
    global camera_running
    camera_running = True
    threading.Thread(target=camera_stream).start()



@socketio.on('stop_camera_stream')
def stop_camera_stream():
    global camera_running
    camera_running = False  # Set the flag to stop the camera
    emit("camera_stopped", {"message": "Camera has been stopped."})  # Optional confirmation message

def camera_stream():
    global camera_running
    cap = cv2.VideoCapture(0)

    while camera_running:  # Keep checking the camera_running flag
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))
        results = model(frame)

        measurements, annotated_frame = process_detections(frame, results, conversion_factor)

        # Send the processed frame and measurements to the client
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_data = buffer.tobytes()
        socketio.emit('frame', {
            'frame': frame_data,
            'measurements': measurements,
            'detected_classes': list(set(item['class_name'] for item in measurements))
        })

    cap.release()



def process_detections(frame, results, conversion_factor):
    measurements = []
    detected_classes = set()

    # Check if results contain detections
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        bboxes = results[0].boxes.xyxy.cpu().numpy()

        for idx, (mask, class_id, bbox) in enumerate(zip(masks, class_ids, bboxes)):
            mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:  # Ensure contours exist
                contour = max(contours, key=cv2.contourArea)
                top_y = min(contour[:, :, 1])
                top_points = [pt[0] for pt in contour if abs(pt[0][1] - top_y) < 5]
                leftmost = min(top_points, key=lambda x: x[0])
                rightmost = max(top_points, key=lambda x: x[0])
                top_width_px = rightmost[0] - leftmost[0]
                top_width_mm = top_width_px / conversion_factor
                class_name = results[0].names[int(class_id)]

                measurements.append({
                    "class_name": class_name,
                    "top_width_mm": top_width_mm
                })

                # Draw bounding box and contours only for specific classes
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if "drill" in class_name.lower():
                    # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    text_position = (leftmost[0], leftmost[1] - 10)
                    cv2.putText(frame, f"{top_width_mm:.2f} mm", text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                detected_classes.add(class_name)

    return measurements, frame

if __name__ == '__main__':
    socketio.run(app, debug=True)
