import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import os
import json
import numpy as np
from PIL import Image

# === CONFIG ===
COCO_DIR = "coco"
IMAGES_DIR = os.path.join(COCO_DIR, "images")
ANNOTATIONS_FILE = os.path.join(COCO_DIR, "result.json")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 16
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# === LOAD COCO ANNOTATIONS ===
print("📁 Loading COCO annotations...")
with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

# Extract categories and create class mapping
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
class_names = [categories[i] for i in sorted(categories.keys())]
num_classes = len(class_names)
category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

print(f"✅ Classes found: {class_names}")
print(f"📊 Number of classes: {num_classes}")

# Create image_id to annotations mapping
image_annotations = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

# Create image_id to filename mapping and resolve paths
image_info = {}
for img in coco_data['images']:
    img_id = img['id']
    file_name = img['file_name']
    
    # Handle different path formats
    if '..' in file_name or file_name.startswith('label-studio'):
        # Extract just the filename from the path
        file_name = os.path.basename(file_name)
    
    # Try to find the actual file
    possible_paths = [
        os.path.join(IMAGES_DIR, file_name),  # Direct in images folder
        os.path.join(COCO_DIR, file_name),    # Direct in coco folder
        file_name,                             # Relative to current directory
    ]
    
    # Also check if the original path exists (relative to coco dir)
    original_path = os.path.join(COCO_DIR, img['file_name'])
    if os.path.exists(original_path):
        possible_paths.insert(0, original_path)
    
    # Find the first path that exists
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path:
        image_info[img_id] = actual_path
    else:
        print(f"⚠️  Warning: Could not find image file: {img['file_name']}")

print(f"📊 Found {len(image_info)} valid images out of {len(coco_data['images'])} total")

# Prepare dataset: (image_path, class_index) pairs
dataset_items = []
skipped = 0
for img_id, annotations in image_annotations.items():
    if img_id in image_info:
        img_path = image_info[img_id]
        # Verify file exists
        if os.path.exists(img_path):
            # Use the first annotation's category (for image classification)
            category_id = annotations[0]['category_id']
            class_idx = category_id_to_index[category_id]
            dataset_items.append((img_path, class_idx))
        else:
            skipped += 1
    else:
        skipped += 1

print(f"📊 Total valid images: {len(dataset_items)}")
if skipped > 0:
    print(f"⚠️  Skipped {skipped} images (files not found)")

if len(dataset_items) == 0:
    print("\n❌ ERROR: No valid images found!")
    print("\nPlease check:")
    print("1. Images are in the 'coco/images/' directory")
    print("2. Or adjust the IMAGES_DIR path to match your actual image location")
    print("3. Check the file_name field in your result.json")
    exit(1)

# Shuffle and split dataset
np.random.seed(123)
np.random.shuffle(dataset_items)
split_idx = int(len(dataset_items) * (1 - VALIDATION_SPLIT))
train_items = dataset_items[:split_idx]
val_items = dataset_items[split_idx:]

print(f"📊 Training samples: {len(train_items)}")
print(f"📊 Validation samples: {len(val_items)}")

# Save class names
os.makedirs(MODEL_DIR, exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)
print(f"📁 Saved class names to {CLASS_NAMES_PATH}")

# === CREATE TF DATASET ===
def load_and_preprocess_image(img_path, label):
    """Load and preprocess a single image"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, label

def create_dataset(items, is_training=True):
    """Create TensorFlow dataset from list of (path, label) tuples"""
    paths = [item[0] for item in items]
    labels = [item[1] for item in items]
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_items, is_training=True)
val_ds = create_dataset(val_items, is_training=False)

# === DATA AUGMENTATION ===
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# === IMPROVED MODEL ARCHITECTURE ===
base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# === COMPILE MODEL ===
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# === CALLBACKS ===
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# === PHASE 1: TRAIN WITH FROZEN BASE ===
print("\n" + "="*60)
print("PHASE 1: Training with frozen base model")
print("="*60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# === PHASE 2: FINE-TUNE BASE MODEL ===
print("\n" + "="*60)
print("PHASE 2: Fine-tuning - Unfreezing base model")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# === SAVE FINAL MODEL ===
model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"📋 Classes: {class_names}")

# === SAVE TRAINING HISTORY ===
combined_history = {
    'accuracy': history.history['accuracy'] + history_fine.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'] + history_fine.history['val_accuracy'],
    'loss': history.history['loss'] + history_fine.history['loss'],
    'val_loss': history.history['val_loss'] + history_fine.history['val_loss']
}

with open(HISTORY_PATH, 'w') as f:
    json.dump(combined_history, f, indent=2)
print(f"📊 Training history saved to {HISTORY_PATH}")

# === EVALUATE ON VALIDATION ===
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

val_loss, val_acc = model.evaluate(val_ds)
print(f"✅ Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"✅ Validation Loss: {val_loss:.4f}")

# Evaluate on training set
train_eval_ds = create_dataset(train_items, is_training=False)
train_loss, train_acc = model.evaluate(train_eval_ds)
print(f"✅ Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"✅ Training Loss: {train_loss:.4f}")

# === INTERPRET RESULTS ===
print("\n" + "="*60)
print("TRAINING ANALYSIS")
print("="*60)

overfitting_gap = train_acc - val_acc
print(f"📈 Training-Validation Gap: {overfitting_gap:.4f}")

if train_acc >= 0.95 and val_acc < 0.75:
    print("⚠️  Model is OVERFITTING: memorized training data but not generalizing well.")
    print("💡 Suggestions: More data augmentation, increase dropout, or collect more data.")
elif train_acc < 0.75 and val_acc < 0.75:
    print("❌ Model is UNDERFITTING: needs more training or model capacity.")
    print("💡 Suggestions: Train longer, reduce regularization, or use larger model.")
elif val_acc >= 0.85:
    print("✅ Model is performing WELL and generalizing properly!")
    if overfitting_gap < 0.1:
        print("✨ Excellent generalization - minimal overfitting!")
else:
    print("🔄 Model is learning but could be improved.")
    print("💡 Consider training longer or adjusting hyperparameters.")

# === PRINT TRAINING SUMMARY ===
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Total Epochs Completed: {len(combined_history['accuracy'])}")
print(f"Best Validation Accuracy: {max(combined_history['val_accuracy']):.4f}")
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Model saved to: {MODEL_PATH}")
print(f"History saved to: {HISTORY_PATH}")

print("\n" + "="*60)
print("TRAINING COMPLETE! 🎉")
print("="*60)
print("\n💡 Next steps:")
print("   1. Run the FastAPI server: uvicorn main:app --reload")
print("   2. Test accuracy: GET http://localhost:8000/test_accuracy")
print("   3. Make predictions: POST http://localhost:8000/predict")