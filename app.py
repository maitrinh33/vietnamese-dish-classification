from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import json
import os
from dotenv import load_dotenv
import base64
from datetime import datetime


app = Flask(__name__)

# Sử dụng đường dẫn tương đối để dễ bảo trì và tương thích với các hệ thống khác
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Config from environment with sane defaults
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(BASE_DIR, "models", "best.pt"))
CLASS_NAMES_PATH = os.getenv('CLASS_NAMES_PATH', os.path.join(BASE_DIR, "class_names.txt"))
INGREDIENTS_PATH = os.getenv('INGREDIENTS_PATH', os.path.join(BASE_DIR, "dish_ingredients.json"))
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.3'))
TOP_K = int(os.getenv('TOP_K', '3'))

# Tải model YOLO và metadata
model = YOLO(MODEL_PATH)
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    CLASS_NAMES = []

if os.path.exists(INGREDIENTS_PATH):
    with open(INGREDIENTS_PATH, "r", encoding="utf-8") as f:
        DISH_TO_INGREDIENTS = json.load(f)
else:
    DISH_TO_INGREDIENTS = {}

# Name Map 
PRETTY_NAME_MAP = {
    'banh_mi': 'Banh Mi',
    'banh_pia': 'Banh Pia',
    'banh_tai_heo': 'Banh Tai Heo',
    'banh_tieu': 'Banh Tieu',
    'banh_trang_nuong': 'Banh Trang Nuong',
    'banh_troi_nuoc': 'Banh Troi Nuoc',
    'banh_trung_thu': 'Banh Trung Thu',
    'banh_u': 'Banh U',
    'banh_xeo': 'Banh Xeo',
    'bo_kho': 'Bo Kho',
    'bo_ne': 'Bo Ne',
    'bo_nuong_la_lot': 'Bo Nuong La Lot',
    'bun_bo_hue': 'Bun Bo Hue',
    'bun_cha': 'Bun Cha',
    'bun_cha_ca': 'Bun Cha Ca',
}

# Thư mục lưu ảnh tải lên
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def _format_display_name(slug: str) -> str:
    return PRETTY_NAME_MAP.get(slug, slug.replace('_', ' ').title())

def predict_image(file_path):
    """Dự đoán món ăn bằng mô hình YOLO classification, trả về (class_name, confidence, ingredients)."""
    try:
        img = Image.open(file_path).convert("RGB")
        result = model.predict(img, imgsz=224, verbose=False)[0]
        probs = result.probs.data.detach().cpu().numpy().astype(float)
        # Top-K indices and confidences
        top_indices = probs.argsort()[::-1][:max(1, TOP_K)]
        top_items = []
        for idx in top_indices:
            name = result.names.get(int(idx), CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else str(int(idx)))
            conf = float(probs[int(idx)])
            top_items.append({
                'dish': _format_display_name(name),
                'confidence': conf
            })

        class_id = int(top_indices[0])
        class_name = result.names.get(class_id, CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id))
        confidence = float(probs[class_id])
        ingredients = DISH_TO_INGREDIENTS.get(class_name, [])
        return _format_display_name(class_name), confidence, ingredients, top_items
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, []

@app.route('/', methods=['GET', 'POST'])
def home():
    """Trang chủ với chức năng tải ảnh và dự đoán."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400  

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Lưu tệp vào thư mục chỉ định
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        print(f"Tệp đã lưu tại: {file_path}")  # Kiểm tra xem tệp đã được lưu

        # Gọi hàm dự đoán
        class_name, confidence, ingredients, top_items = predict_image(file_path)
        if class_name is not None:
            return jsonify({
                'dish': class_name,
                'confidence': confidence,
                'ingredients': ingredients,
                'topk': top_items,
                'file_path': url_for('download_file', filename=file.filename)  # Đường dẫn tải về
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500  # Dự đoán thất bại

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Phục vụ tệp đã tải lên từ thư mục UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)

# -----------------------------
# Webcam streaming endpoints
# -----------------------------
def _annotate_frame(frame):
    """Run model on BGR frame and overlay class + confidence + first 3 ingredients."""
    result = model.predict(frame, imgsz=224, verbose=False)[0]
    probs = result.probs.data.detach().cpu().numpy().astype(float)
    class_id = int(probs.argmax())
    class_name = result.names.get(class_id, CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id))
    confidence = float(probs[class_id])
    text = f"{_format_display_name(class_name)} {confidence*100:.1f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 180, 120), 2, cv2.LINE_AA)
    ings = DISH_TO_INGREDIENTS.get(class_name, [])[:3]
    if ings:
        cv2.putText(frame, ", ".join(ings), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 120, 255), 2, cv2.LINE_AA)
    return frame

def _gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở camera")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _annotate_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/cam')
def cam_page():
    return render_template('cam.html')

@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config')
def get_config():
    return jsonify({
        'model_path': MODEL_PATH,
        'class_names_path': CLASS_NAMES_PATH,
        'ingredients_path': INGREDIENTS_PATH,
        'conf_threshold': CONF_THRESHOLD,
        'top_k': TOP_K
    })

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Accept a base64 data URL image from the camera page, save, and return prediction JSON."""
    data = request.get_json(silent=True) or {}
    data_url = data.get('image')
    if not data_url or not isinstance(data_url, str) or 'base64,' not in data_url:
        return jsonify({'error': 'Invalid image data'}), 400

    header, b64 = data_url.split('base64,', 1)
    try:
        binary = base64.b64decode(b64)
    except Exception:
        return jsonify({'error': 'Decode failed'}), 400

    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"cam_{ts}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, 'wb') as f:
        f.write(binary)

    class_name, confidence, ingredients, top_items = predict_image(file_path)
    if class_name is None:
        return jsonify({'error': 'Prediction failed'}), 500

    return jsonify({
        'dish': class_name,
        'confidence': confidence,
        'ingredients': ingredients,
        'topk': top_items,
        'file_path': url_for('download_file', filename=filename)
    })

if __name__ == '__main__':
    app.run(debug=True)

