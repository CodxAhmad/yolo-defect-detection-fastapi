from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# -------------------------
# App initialization
# -------------------------
app = FastAPI(title="Defective Product Detection API")

# -------------------------
# CORS (allow all for now)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load model ONCE
# -------------------------
try:
    model = YOLO("models/best.pt")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

CONF_THRESHOLD = 0.5
DAMAGED_CLASS_NAME = "damaged"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ---- Validate file name ----
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload jpg, jpeg, or png"
        )

    # ---- Read image ----
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # ---- Run inference ----
    try:
        results = model(img_np, conf=CONF_THRESHOLD)
    except Exception:
        raise HTTPException(status_code=500, detail="Model inference failed")

    # ---- Draw damaged boxes only ----
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = r.names[cls_id]

            if class_name != DAMAGED_CLASS_NAME:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img_np,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    # ---- Encode output image ----
    success, encoded_img = cv2.imencode(".jpg", img_np)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    return StreamingResponse(
        io.BytesIO(encoded_img.tobytes()),
        media_type="image/jpeg"
    )
