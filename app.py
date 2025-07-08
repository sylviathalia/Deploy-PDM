from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model = YOLO("model/best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run detection
    results = model(img)

    # Ambil nama kelas (label) dan koordinat box
    names = results[0].names
    boxes = []
    classes = []

    for box in results[0].boxes:
        xyxy = box.xyxy.cpu().numpy().tolist()[0]  # x1, y1, x2, y2
        cls = int(box.cls.cpu().numpy()[0])        # class index
        boxes.append(xyxy)
        classes.append(names[cls])

    # Return sebagai JSON
    return jsonify({'labels': classes, 'boxes': boxes})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
