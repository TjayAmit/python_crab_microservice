from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras import applications
import numpy as np
from PIL import Image
import io
import logging
import json
import os
from typing import List

from models_difference_result import (
    get_model_comparison,
    get_single_model_info,
    get_quick_summary
)

app = FastAPI(title="Crab Classification API")

# === CONFIG ===
# Prefer the cropped/scale-robust model (train_model_cropped.py). The cropped
# pipeline now trains with randomized framing (tight crop -> full frame), so the
# whole-photo input this API sends matches what the model saw during training.
CLASS_NAMES_PATH = "model/class_names.json"
IMAGE_SIZE = (224, 224)  # Updated to match training (MobileNetV2 default)
COCO_DIR = "coco"
IMAGES_DIR = os.path.join(COCO_DIR, "images")

# First existing path wins; lets the app run before and after a retrain.
MODEL_CANDIDATES = [
    "model/my_model_cropped.keras",          # final model from train_model_cropped.py
    "model/checkpoint_best_cropped.keras",   # best checkpoint from the cropped run
    "model/my_model.keras",                  # previous cropped run
    "model/best_model.keras",
]

# === LOAD MODEL ===
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        f"❌ No model found. Looked for: {MODEL_CANDIDATES}"
    )

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
    """
    Convert uploaded image to model-ready tensor with MobileNetV2 preprocessing
    CRITICAL: Must match training preprocessing exactly!
    """
    try:
        # Load and convert to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to model's expected input size
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Apply MobileNetV2 preprocessing (scales to [-1, 1] range)
        img_array = applications.mobilenet_v2.preprocess_input(img_array)
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# === PREDICTION WITH CONFIDENCE THRESHOLD ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded crab image
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
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = {
            CLASS_NAMES[int(idx)]: float(probabilities[idx])
            for idx in top_3_indices
        }
        
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

        # Determine confidence level and provide guidance
        if top_confidence >= 0.90:
            confidence_level = "Very High"
            guidance = "Model is very confident in this prediction."
        elif top_confidence >= 0.75:
            confidence_level = "High"
            guidance = "Model is confident in this prediction."
        elif top_confidence >= 0.60:
            confidence_level = "Moderate"
            guidance = "Model has moderate confidence. Consider checking other top predictions."
        elif top_confidence >= 0.40:
            confidence_level = "Low"
            guidance = "Model has low confidence. Multiple classes are possible."
        else:
            confidence_level = "Very Low"
            guidance = "Model is uncertain. Image may be unclear or not match training data."

        return JSONResponse({
            "status": "success",
            "predicted_label": top_label,
            "confidence": round(top_confidence, 4),
            "confidence_level": confidence_level,
            "guidance": guidance,
            "top_3_predictions": top_3_predictions,
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
    Predict classes for multiple uploaded crab images
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
                
                # Get top 3 for each image
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                top_3 = {
                    CLASS_NAMES[int(idx)]: round(float(probabilities[idx]), 4)
                    for idx in top_3_indices
                }
                
                results.append({
                    "filename": file.filename,
                    "predicted_label": top_label,
                    "confidence": round(top_confidence, 4),
                    "top_3_predictions": top_3,
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
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        })
        
    except Exception as e:
        logging.error(f"[predict_batch] Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === DATASET SUMMARY ===
@app.get("/dataset_summary")
async def dataset_summary():
    """
    Get summary statistics of the training dataset from COCO annotations
    """
    try:
        annotations_file = os.path.join(COCO_DIR, "result.json")
        
        if not os.path.exists(annotations_file):
            return JSONResponse(
                {"error": f"Annotations file not found at {annotations_file}"},
                status_code=404
            )

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Count images per category
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        class_counts = {name: 0 for name in categories.values()}
        
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            cat_name = categories[cat_id]
            class_counts[cat_name] += 1

        total_images = len(coco_data['images'])
        total_annotations = len(coco_data['annotations'])

        return JSONResponse({
            "status": "success",
            "dataset_dir": COCO_DIR,
            "total_images": total_images,
            "total_annotations": total_annotations,
            "num_classes": len(categories),
            "classes": list(categories.values()),
            "class_distribution": class_counts
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
        # Get model architecture info
        total_params = model.count_params()
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        return JSONResponse({
            "status": "success",
            "model_path": MODEL_PATH,
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "total_parameters": total_params,
            "preprocessing": "MobileNetV2 (values scaled to [-1, 1])",
            "model_loaded": model is not None
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# === TEST ACCURACY ON VALIDATION IMAGES ===
@app.get("/test_accuracy")
async def test_accuracy():
    """
    Test model accuracy on sample images from the dataset
    """
    try:
        annotations_file = os.path.join(COCO_DIR, "result.json")
        
        if not os.path.exists(annotations_file):
            return JSONResponse(
                {"error": f"Annotations file not found at {annotations_file}"},
                status_code=404
            )

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Build category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Build image info
        image_info = {}
        for img in coco_data['images']:
            img_id = img['id']
            file_name = img['file_name']
            
            if '..' in file_name or file_name.startswith('label-studio'):
                file_name = os.path.basename(file_name)
            
            possible_paths = [
                os.path.join(IMAGES_DIR, file_name),
                os.path.join(COCO_DIR, file_name),
                file_name,
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    image_info[img_id] = path
                    break

        # Get annotations
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # Sample images for testing (up to 5 per class)
        class_samples = {cat_name: [] for cat_name in categories.values()}
        
        for img_id, annotations in image_annotations.items():
            if img_id not in image_info:
                continue
                
            cat_id = annotations[0]['category_id']
            cat_name = categories[cat_id]
            
            if len(class_samples[cat_name]) < 5:
                class_samples[cat_name].append((img_id, image_info[img_id], cat_name))

        # Test predictions
        results = []
        correct_count = 0
        total_count = 0
        class_correct = {cat_name: 0 for cat_name in categories.values()}
        class_total = {cat_name: 0 for cat_name in categories.values()}
        
        for cat_name, samples in class_samples.items():
            for img_id, img_path, true_label in samples:
                total_count += 1
                class_total[true_label] += 1
                
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
                    
                    correct = pred_label == true_label
                    if correct:
                        correct_count += 1
                        class_correct[true_label] += 1

                    # Get top 3 predictions
                    top_3_indices = np.argsort(probabilities)[-3:][::-1]
                    top_3 = {
                        CLASS_NAMES[int(idx)]: round(float(probabilities[idx]), 4)
                        for idx in top_3_indices
                    }

                    results.append({
                        "image": os.path.basename(img_path),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": round(confidence, 4),
                        "correct": correct,
                        "top_3_predictions": top_3
                    })
                    
                except Exception as e:
                    results.append({
                        "image": os.path.basename(img_path),
                        "true_label": true_label,
                        "error": str(e),
                        "correct": False
                    })

        # Calculate accuracies
        overall_accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0
        
        per_class_accuracy = {}
        for cat_name in categories.values():
            if class_total[cat_name] > 0:
                acc = class_correct[cat_name] / class_total[cat_name]
                per_class_accuracy[cat_name] = {
                    "accuracy": round(acc, 4),
                    "correct": class_correct[cat_name],
                    "total": class_total[cat_name]
                }

        return JSONResponse({
            "status": "success",
            "total_images_tested": total_count,
            "correct_predictions": correct_count,
            "overall_accuracy": overall_accuracy,
            "accuracy_percentage": f"{overall_accuracy * 100:.2f}%",
            "per_class_accuracy": per_class_accuracy,
            "sample_results": results
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
        "message": "Crab Classification API",
        "version": "3.0",
        "model": "MobileNetV2 Transfer Learning",
        "classes": CLASS_NAMES,
        "endpoints": {
            "POST /predict": "Predict single crab image",
            "POST /predict_batch": "Predict multiple crab images",
            "GET /test_accuracy": "Test model on sample images",
            "GET /dataset_summary": "Get dataset statistics",
            "GET /model_info": "Get model information",
            "GET /health": "Health check"
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
        "model_path": MODEL_PATH,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE
    }

# === Model Comparison ===
@app.get("/difference-of-training-models")
async def comparison_endpoint():
    return get_model_comparison()

@app.get("/model/{model_name}")
async def single_model_endpoint(model_name: str):
    return get_single_model_info(model_name)

@app.get("/models/summary")
async def summary_endpoint():
    return get_quick_summary()