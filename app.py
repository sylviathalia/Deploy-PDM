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

    # Ambil dict label dari model
    label_dict = model.names

    boxes = []
    classes = []

    for box in results[0].boxes:
        xyxy = box.xyxy.cpu().numpy().tolist()[0]
        cls_idx = int(box.cls.cpu().numpy()[0])
        cls_name = label_dict[cls_idx]
        boxes.append(xyxy)
        classes.append(cls_name)

    return jsonify({'labels': classes, 'boxes': boxes})
