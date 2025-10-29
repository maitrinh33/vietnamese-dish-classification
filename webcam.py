import cv2
from ultralytics import YOLO
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
ING_PATH = os.path.join(BASE_DIR, "dish_ingredients.json")

model = YOLO(MODEL_PATH)

try:
    with open(ING_PATH, "r", encoding="utf-8") as f:
        DISH_TO_ING = json.load(f)
except Exception:
    DISH_TO_ING = {}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(frame, imgsz=224, verbose=False)[0]
    cls_id = int(res.probs.top1)
    cls_name = res.names[cls_id]
    conf = float(res.probs.top1conf)

    text = f"{cls_name} {conf*100:.1f}%"
    cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 180, 120), 2, cv2.LINE_AA)

    # Show up to first 3 ingredients as a hint
    ings = DISH_TO_ING.get(cls_name, [])[:3]
    if ings:
        cv2.putText(frame, ", ".join(ings), (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 120, 255), 2, cv2.LINE_AA)

    cv2.imshow("VN Food - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


