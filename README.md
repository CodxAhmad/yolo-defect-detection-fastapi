# Defective Image Detection using YOLOv8 + FastAPI

This project detects **damaged/defective regions** in images using **YOLOv8 (Ultralytics)** and serves predictions via a **FastAPI REST API**.

The model is trained using **transfer learning** on a custom dataset with bounding box annotations and performs **object detection** (not classification).

---

## 📌 Features

- YOLOv8 object detection (transfer learning)
- Automatic preprocessing & augmentation via Ultralytics
- Bounding box visualization for detected defects
- REST API for inference using FastAPI
- Health check endpoint for deployment readiness
- Ready for Docker & cloud deployment

---

## 📁 Project Structure

├── app/
│ └── main.py # FastAPI application
├── data/
│ ├── images/
│ │ ├── train/
│ │ ├── val/
│ │ └── test/
│ └── labels/
│ ├── train/
│ ├── val/
│ └── test/
├── runs/
│ └── detect/
│ └── train/
│ └── weights/
│ └── best.pt
├── scripts/
│ └── csv_to_yolo.py # CSV → YOLO label conversion
├── data.yaml # YOLO dataset config
├── requirements.txt
├── .gitignore
└── README.md


---

## 📊 Dataset & Annotation Format

- Original annotations were stored in **CSV format**
- Converted into **YOLO format**:

<class_id> <x_center> <y_center> <width> <height>


All values are **normalized (0–1)** relative to image width & height.

---

## 🧠 Model Training (YOLOv8)

Install Ultralytics:
pip install ultralytics

Train the model:

    yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640


YOLOv8 automatically handles:

Image resizing

Label scaling

Data augmentation

Loss computation

Transfer learning

🚀 Running the FastAPI Server
uvicorn app.main:app --reload


API will be available at:

http://127.0.0.1:8000

🔍 API Endpoints
Health Check
GET /health


Response

{
  "status": "ok",
  "model_loaded": true
}

Predict Defects
POST /predict


Request

multipart/form-data

Upload an image file

Response

Image with detected damaged regions highlighted

⚙️ Confidence Threshold

Detection confidence threshold is configurable inside main.py:

CONF_THRESHOLD = 0.5

🛠 Tech Stack

Python

YOLOv8 (Ultralytics)

OpenCV

FastAPI

NumPy

Pillow

📌 Notes

Bounding boxes remain correct under brightness/contrast based augmentations

Model weights are loaded once at startup for efficiency

Designed for easy Dockerization and CI/CD integration

📜 License

This project is for academic and learning purposes.