from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load model
model = YOLO("model/best.pt")  # Pastikan path dan file sudah benar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Baca file gambar dari form
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Prediksi
    results = model(img)

    # Cek apakah ada boxes
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Ambil nama label
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        # Ambil koordinat box
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

        # Buat gambar hasil anotasi
        annotated = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'result': encoded_img, 'labels': labels, 'boxes': boxes, 'message': 'Objects detected!'})
    else:
        # Tidak ada deteksi
        return jsonify({'result': None, 'labels': [], 'boxes': [], 'message': 'No objects detected.'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
