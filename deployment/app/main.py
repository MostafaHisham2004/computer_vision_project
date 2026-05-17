from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import os
import numpy as np
from app.models import YoloService, SegmentationService, ClassificationService

from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to models
BASE_DIR = "/home/mostafaosman/Downloads/Yolo/notebooks"
YOLO_PATH = os.path.join(BASE_DIR, "Yolo/best.pt")
UNET_PATH = os.path.join(BASE_DIR, "UNet/best_deeplabv3_cityscapes.pth")
CNN_PATH = os.path.join(BASE_DIR, "CNN/plant_disease_model.h5")

# Initialize Services as None
yolo_service = None
unet_service = None
cnn_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup (model loading) and shutdown.
    """
    global yolo_service, unet_service, cnn_service
    
    logging.info("Application startup: Loading AI models...")
    
    # Load YOLO Service
    if os.path.exists(YOLO_PATH):
        try:
            yolo_service = YoloService(YOLO_PATH)
            logging.info("YOLO Service initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize YOLO Service: {e}")
    else:
        logging.warning(f"YOLO model not found at {YOLO_PATH}")

    # Load UNet Service
    if os.path.exists(UNET_PATH):
        try:
            unet_service = SegmentationService(UNET_PATH)
            logging.info("UNet Service initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize UNet Service: {e}")
            # We raise here because the user's request is failing specifically on this
            raise RuntimeError(f"Critical failure loading UNet model: {e}")
    else:
        logging.warning(f"UNet model not found at {UNET_PATH}")

    # Load CNN Service
    if os.path.exists(CNN_PATH):
        try:
            cnn_service = ClassificationService(CNN_PATH)
            logging.info("CNN Service initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize CNN Service: {e}")
            import traceback
            logging.error(traceback.format_exc())
    else:
        logging.warning(f"CNN model not found at {CNN_PATH}")

    yield
    
    logging.info("Application shutdown: Cleaning up resources...")
    # Add any necessary cleanup here (e.g., closing database connections)

app = FastAPI(title="Multi-Model AI API", lifespan=lifespan)

@app.post("/predict/yolo")
async def predict_yolo(file: UploadFile = File(...)):
    if yolo_service is None:
        return JSONResponse(status_code=503, content={"error": "YOLO Service is not available"})
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = yolo_service.predict(image)
    
    # Format YOLO result
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    classes = result.boxes.cls.cpu().numpy().tolist()
    scores = result.boxes.conf.cpu().numpy().tolist()
    
    detections = []
    for box, cls, score in zip(boxes, classes, scores):
        detections.append({
            "bbox": box,
            "class": yolo_service.class_names[int(cls)],
            "score": score
        })
    
    return {"detections": detections}

@app.post("/predict/unet")
async def predict_unet(file: UploadFile = File(...)):
    if unet_service is None:
        return JSONResponse(status_code=503, content={"error": "UNet Service is not available"})
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    mask = unet_service.predict(image)
    return {"mask": mask.tolist()}

@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    if cnn_service is None:
        return JSONResponse(status_code=503, content={"error": "CNN Service is not available"})
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    label, confidence = cnn_service.predict(image)
    return {"label": label, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
