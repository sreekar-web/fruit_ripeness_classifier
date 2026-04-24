import os
import io
import sys
import json
import logging
import datetime
import numpy as np
import matplotlib.cm as cm
import boto3
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fruit_ripeness_classifier"))
from src.model import FruitRipenessClassifier
from src.gradcam import GradCAM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fruit Ripeness Classifier API")

CLASS_NAMES = ["ripe", "rotten", "unripe"]

S3_BUCKET   = os.environ.get("S3_BUCKET", "")
MODEL_KEY   = os.environ.get("MODEL_KEY", "models/best_model.pth")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "fruit-predictions")
AWS_REGION  = os.environ.get("AWS_REGION", "us-east-1")

MODEL_PATH  = "/tmp/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = None

def download_model_from_s3():
    """Download model weights from S3 to /tmp on pod startup."""
    logger.info(f"Downloading model from s3://{S3_BUCKET}/{MODEL_KEY}")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, MODEL_KEY, MODEL_PATH)
    logger.info("Model downloaded successfully.")

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        download_model_from_s3()
    m = FruitRipenessClassifier(num_classes=3).to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    model = m
    logger.info("Model loaded and ready.")

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def log_prediction_to_dynamo(filename: str, predicted: str, confidence: float, all_scores: dict):
    """Log every prediction to DynamoDB for AWS SDK visibility."""
    try:
        dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
        table  = dynamo.Table(DYNAMO_TABLE)
        table.put_item(Item={
            "prediction_id": f"{filename}-{datetime.datetime.utcnow().isoformat()}",
            "filename":      filename,
            "predicted":     predicted,
            "confidence":    str(round(confidence * 100, 2)),
            "all_scores":    {k: str(round(v * 100, 2)) for k, v in all_scores.items()},
            "timestamp":     datetime.datetime.utcnow().isoformat()
        })
        logger.info(f"Logged prediction to DynamoDB: {predicted} ({confidence:.2%})")
    except Exception as e:
        logger.warning(f"DynamoDB logging failed (non-fatal): {e}")

def overlay_heatmap(original_img: Image.Image, cam: np.ndarray) -> str:
    """Overlay Grad-CAM heatmap on original image, return as base64 string."""
    heatmap      = cm.jet(cam)[:, :, :3]
    heatmap      = np.uint8(255 * heatmap)
    heatmap_img  = Image.fromarray(heatmap).resize((224, 224))
    original_arr = np.array(original_img.resize((224, 224)), dtype=np.float32)
    heatmap_arr  = np.array(heatmap_img, dtype=np.float32)
    overlay      = np.uint8(np.clip(0.5 * original_arr + 0.5 * heatmap_arr, 0, 255))
    overlay_img  = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
def health():
    """Kubernetes liveness probe endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image, return predicted ripeness class + confidence scores.
    Logs result to DynamoDB.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    input_tensor = get_transform()(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    predicted_idx   = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(probs[predicted_idx])
    all_scores      = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    log_prediction_to_dynamo(file.filename, predicted_class, confidence, all_scores)

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence":      round(confidence * 100, 2),
        "all_scores": {k: round(v * 100, 2) for k, v in all_scores.items()}
    })

@app.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Same as /predict but also returns a base64 Grad-CAM overlay image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    input_tensor = get_transform()(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

    predicted_idx   = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(probs[predicted_idx])
    all_scores      = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    target_layer = model.block3[0]
    gradcam      = GradCAM(model, target_layer)
    cam, _       = gradcam.generate(input_tensor.to(device))
    overlay_b64  = overlay_heatmap(image, cam)

    log_prediction_to_dynamo(file.filename, predicted_class, confidence, all_scores)

    return JSONResponse({
        "predicted_class":  predicted_class,
        "confidence":       round(confidence * 100, 2),
        "all_scores":       {k: round(v * 100, 2) for k, v in all_scores.items()},
        "gradcam_overlay":  overlay_b64
    })