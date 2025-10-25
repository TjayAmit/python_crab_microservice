from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import json
import os
from typing import List

app = FastAPI(title="Curacha Classification API")

# === CONFIG ===
MODEL_PATH = "model/best_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMAGE_SIZE = (180, 180)
DATASET_DIR = "curacha_dataset"

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# === LOAD CLASS NAMES ===
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    print(f"✅ Loaded class names: {CLASS_NAMES}")
else:
    raise FileNotFoundError(f"❌ class_names.json not found at {CLASS_NAMES_PATH}")

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    filename="prediction_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === IMAGE PREPROCESSING ===
def preprocess_image(image_bytes):
    """Convert uploaded image to model-ready tensor with proper preprocessing"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)  # High-quality resizing
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# === PREDICTION WITH CONFIDENCE THRESHOLD ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image
    Returns predicted label, confidence, and all class probabilities
    """
    try:
        contents = await file.read()
        img_tensor = preprocess_image(contents)

        # Get predictions
        preds = model.predict(img_tensor, verbose=0)
        probabilities = preds[0]
        
        # Get top prediction
        top_idx = int(np.argmax(probabilities))
        top_label = CLASS_NAMES[top_idx]
        top_confidence = float(probabilities[top_idx])
        
        # Get all class probabilities
        all_predictions = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort predictions by confidence
        sorted_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )

        # Log prediction
        logging.info(
            f"[predict] File: {file.filename} | "
            f"Predicted: {top_label} | Confidence: {top_confidence:.4f}"
        )

        # Determine confidence level
        if top_confidence >= 0.9:
            confidence_level = "Very High"
        elif top_confidence >= 0.75:
            confidence_level = "High"
        elif top_confidence >= 0.6:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        return JSONResponse({
            "status": "success",
            "predicted_label": top_label,
            "confidence": round(top_confidence, 4),
            "confidence_level": confidence_level,
            "all_predictions": sorted_predictions,
            "filename": file.filename
        })

    except ValueError as ve:
        logging.error(f"[predict] Validation error: {str(ve)}")
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        logging.error(f"[predict] Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === BATCH PREDICTION ===
@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict classes for multiple uploaded images
    """
    try:
        results = []
        
        for file in files:
            try:
                contents = await file.read()
                img_tensor = preprocess_image(contents)
                
                preds = model.predict(img_tensor, verbose=0)
                probabilities = preds[0]
                
                top_idx = int(np.argmax(probabilities))
                top_label = CLASS_NAMES[top_idx]
                top_confidence = float(probabilities[top_idx])
                
                results.append({
                    "filename": file.filename,
                    "predicted_label": top_label,
                    "confidence": round(top_confidence, 4),
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "failed"
                })
        
        return JSONResponse({
            "status": "success",
            "total_files": len(files),
            "results": results
        })
        
    except Exception as e:
        logging.error(f"[predict_batch] Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === DATASET SUMMARY ===
@app.get("/dataset_summary")
async def dataset_summary():
    """
    Get summary statistics of the training dataset
    """
    try:
        if not os.path.exists(DATASET_DIR):
            return JSONResponse(
                {"error": f"Dataset directory '{DATASET_DIR}' not found."},
                status_code=404
            )

        summary = {}
        total_images = 0

        for class_name in os.listdir(DATASET_DIR):
            class_path = os.path.join(DATASET_DIR, class_name)
            if os.path.isdir(class_path):
                image_files = [
                    f for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                ]
                count = len(image_files)
                summary[class_name] = count
                total_images += count

        return JSONResponse({
            "status": "success",
            "dataset_dir": DATASET_DIR,
            "total_images": total_images,
            "num_classes": len(summary),
            "class_summary": summary
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# === MODEL INFO ===
@app.get("/model_info")
async def model_info():
    """
    Get information about the loaded model
    """
    try:
        return JSONResponse({
            "status": "success",
            "model_path": MODEL_PATH,
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "model_loaded": model is not None
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# === TEST ACCURACY ON KNOWN IMAGES ===
@app.get("/test_accuracy")
async def test_accuracy():
    """
    Test model accuracy on sample images from each class
    """
    try:
        results = []
        correct_count = 0
        total_count = 0
        
        # Dynamically find test images from dataset
        for class_name in CLASS_NAMES:
            class_path = os.path.join(DATASET_DIR, class_name)
            
            if not os.path.exists(class_path):
                continue
                
            # Get up to 3 images per class
            image_files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ][:3]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                total_count += 1
                
                try:
                    # Read and preprocess
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    img_tensor = preprocess_image(img_bytes)

                    # Predict
                    preds = model.predict(img_tensor, verbose=0)
                    probabilities = preds[0]
                    pred_idx = int(np.argmax(probabilities))
                    pred_label = CLASS_NAMES[pred_idx]
                    confidence = float(probabilities[pred_idx])
                    
                    correct = pred_label == class_name
                    if correct:
                        correct_count += 1

                    # Get all predictions for this image
                    all_preds = {
                        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
                        for i in range(len(CLASS_NAMES))
                    }

                    results.append({
                        "image": img_file,
                        "true_label": class_name,
                        "predicted_label": pred_label,
                        "confidence": round(confidence, 4),
                        "correct": correct,
                        "all_predictions": all_preds
                    })
                    
                except Exception as e:
                    results.append({
                        "image": img_file,
                        "true_label": class_name,
                        "error": str(e),
                        "correct": False
                    })

        accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0

        return JSONResponse({
            "status": "success",
            "total_images": total_count,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "accuracy_percentage": f"{accuracy * 100:.2f}%",
            "results": results
        })

    except Exception as e:
        logging.error(f"[test_accuracy] Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === ROOT ENDPOINT ===
@app.get("/")
async def root():
    """
    API root endpoint with available routes
    """
    return {
        "message": "Curacha Classification API",
        "version": "2.0",
        "endpoints": {
            "POST /predict": "Predict single image",
            "POST /predict_batch": "Predict multiple images",
            "GET /test_accuracy": "Test model on sample images",
            "GET /dataset_summary": "Get dataset statistics",
            "GET /model_info": "Get model information"
        }
    }

# === HEALTH CHECK ===
@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    }