from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
model = YOLO("model/best.pt")  # Pastikan path model sudah benar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img)

    # Jika tidak ada box terdeteksi, results[0].boxes akan None
    if results[0].boxes is None:
        return jsonify({'result': None, 'labels': [], 'boxes': []})

    # Ambil label
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
    # Ambil koordinat box
    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

    # Buat gambar beranotasi
    annotated = results[0].plot()

    # Encode ke base64 supaya bisa ditampilkan di web
    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'result': encoded_img, 'labels': labels, 'boxes': boxes})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
