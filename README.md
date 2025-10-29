
# Vietnamese Food Classifier

A deep learning-based web application for classifying Vietnamese dishes using YOLOv8. Upload images or use your webcam to identify 15 popular Vietnamese dishes and view their ingredients.

## 🍜 Features

- **Image Classification**: Upload images to identify Vietnamese dishes
- **Real-time Webcam Detection**: Live camera feed with on-screen predictions
- **High Accuracy**: 94.2% top-1 accuracy on 15 dish categories 
- **Ingredient Information**: Automatically displays suggested ingredients for each dish
- **Modern UI**: Clean, responsive design with real-time feedback
- **Fast Inference**: ~2.7ms per image on GPU, suitable for real-time use

## 📋 Supported Dishes

- Banh Mi 
- Banh Pia
- Banh Tai Heo
- Banh Tieu
- Banh Trang Nuong 
- Banh Troi Nuoc 
- Banh Trung Thu 
- Banh U
- Banh Xeo 
- Bo Kho 
- Bo Ne 
- Bo Nuong La Lot 
- Bun Bo Hue 
- Bun Cha 
- Bun Cha Ca 

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Webcam (for real-time detection feature)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/maitrinh33/vietnamese-dish-classification.git
   cd vietnamese-dish-classification
   ```

2. **Install dependencies**
   ```bash
   pip freeze > requirements.txt
   pip install -r requirements.txt
   pip install ultralytics opencv-python pillow python-dotenv
   ```

3. **Setup model files**
   - Ensure `models/best.pt` is present (your trained YOLOv8 model)
   - Verify `class_names.txt` contains the 15 dish names
   - Check `dish_ingredients.json` for ingredient mappings

4. **Configure (Optional)**
   Create a `.env` file for custom paths:
   ```env
   MODEL_PATH=models/best.pt
   CLASS_NAMES_PATH=class_names.txt
   INGREDIENTS_PATH=dish_ingredients.json
   CONF_THRESHOLD=0.3
   TOP_K=3
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   - Open your browser to `http://localhost:5000`
   - Upload an image or click "Open Camera" for real-time detection

## 📁 Project Structure

```
train_food2/
├── app.py                 # Flask application server
├── models/
│   └── best.pt           # Trained YOLOv8 model (3MB)
├── templates/
│   ├── index.html        # Main web interface
│   └── cam.html          # Webcam interface
├── uploads/              # User-uploaded images storage
├── class_names.txt       # Class label mapping
├── dish_ingredients.json # Dish-to-ingredients mapping
├── requirements.txt      # Python dependencies
├── webcam.py            # Standalone webcam script (optional)
└── .env                  # Configuration file (optional)
```

## 🎯 Usage

### Image Upload
1. Click "Choose Image" button
2. Select a Vietnamese dish image (JPG, PNG supported)
3. View prediction results with confidence score and ingredients

### Real-time Webcam
1. Click "Open Camera" button
2. Position food in camera view
3. Click "Capture" to analyze current frame
4. Results appear below with dish name, confidence, and ingredients

## 🔧 Technical Details

- **Model**: YOLOv8n-cls (Nano variant)
- **Architecture**: CSPDarknet backbone with 1.46M parameters
- **Input Size**: 224×224 pixels
- **Accuracy**: 94.2% top-1, 99.6% top-5
- **Framework**: PyTorch 2.8+, Ultralytics YOLOv8
- **Backend**: Flask 3.1+
- **Frontend**: HTML5, CSS3, JavaScript

## 📊 Training Information

- **Dataset**: 1,834 images (15 classes)
- **Split**: 70% train (1,283) / 30% val (551)
- **Epochs**: 20
- **Training Time**: ~7-8 minutes on Tesla T4 GPU
- **Platform**: Google Colab

## 🛠️ Development

### Training the Model

The model was trained using Ultralytics YOLOv8 on Google Colab:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(data='/path/to/dataset', epochs=20, imgsz=224, batch=32)
```

### Adding New Dishes

1. Add dish images to `dataset/train/<dish_name>/` and `dataset/val/<dish_name>/`
2. Update `class_names.txt` with new dish name
3. Add ingredient mapping to `dish_ingredients.json`
4. Retrain the model with updated dataset
5. Update `PRETTY_NAME_MAP` in `app.py` for display formatting

## 📝 Notes

- Best results with full dish visibility
- GPU recommended for webcam real-time detection
- Images are saved to `uploads/` directory after upload/capture

---



