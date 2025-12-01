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
IMAGE_SIZE = (224, 224)  # Increased for better feature extraction
BATCH_SIZE = 32  # Increased for more stable gradients
EPOCHS = 100  # More epochs for convergence
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
        file_name = os.path.basename(file_name)
    
    # Try to find the actual file
    possible_paths = [
        os.path.join(IMAGES_DIR, file_name),
        os.path.join(COCO_DIR, file_name),
        file_name,
    ]
    
    original_path = os.path.join(COCO_DIR, img['file_name'])
    if os.path.exists(original_path):
        possible_paths.insert(0, original_path)
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path:
        image_info[img_id] = actual_path

print(f"📊 Found {len(image_info)} valid images out of {len(coco_data['images'])} total")

# Prepare dataset and count per class
dataset_items = []
class_counts = {i: 0 for i in range(num_classes)}
skipped = 0

for img_id, annotations in image_annotations.items():
    if img_id in image_info:
        img_path = image_info[img_id]
        if os.path.exists(img_path):
            category_id = annotations[0]['category_id']
            class_idx = category_id_to_index[category_id]
            dataset_items.append((img_path, class_idx))
            class_counts[class_idx] += 1
        else:
            skipped += 1
    else:
        skipped += 1

print(f"📊 Total valid images: {len(dataset_items)}")
print(f"\n📊 Images per class:")
for idx, name in enumerate(class_names):
    print(f"   {name}: {class_counts[idx]} images")

if skipped > 0:
    print(f"\n⚠️  Skipped {skipped} images (files not found)")

if len(dataset_items) == 0:
    print("\n❌ ERROR: No valid images found!")
    exit(1)

# Shuffle and split dataset
np.random.seed(123)
np.random.shuffle(dataset_items)
split_idx = int(len(dataset_items) * (1 - VALIDATION_SPLIT))
train_items = dataset_items[:split_idx]
val_items = dataset_items[split_idx:]

print(f"\n📊 Training samples: {len(train_items)}")
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
        dataset = dataset.shuffle(2000)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_items, is_training=True)
val_ds = create_dataset(val_items, is_training=False)

# === AGGRESSIVE DATA AUGMENTATION ===
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.4),  # More rotation
    layers.RandomZoom(0.3),  # More zoom
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
    layers.RandomTranslation(0.2, 0.2),  # Add translation
], name='data_augmentation')

# === IMPROVED MODEL WITH EFFICIENTNET ===
# EfficientNetB0 is better than MobileNetV2 for classification tasks
base_model = keras.applications.EfficientNetB0(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze base model from the start since we have enough data
base_model.trainable = True

# Freeze only the first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"🔓 Trainable layers in base model: {sum([1 for layer in base_model.layers if layer.trainable])}")

model = keras.Sequential([
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.Rescaling(1./255),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax', name='output')
], name='crab_classifier')

# === COMPILE MODEL WITH BETTER OPTIMIZER ===
initial_learning_rate = 0.001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()
print(f"\n🔢 Total parameters: {model.count_params():,}")
print(f"🔢 Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# === ADVANCED CALLBACKS ===
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,  # More patience
    restore_best_weights=True,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Cosine decay with warmup
def lr_schedule(epoch, lr):
    if epoch < 10:  # Warmup
        return initial_learning_rate * (epoch + 1) / 10
    else:  # Cosine decay
        return initial_learning_rate * 0.5 * (1 + np.cos(np.pi * (epoch - 10) / (EPOCHS - 10)))

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)

# === SINGLE PHASE TRAINING ===
print("\n" + "="*60)
print("TRAINING WITH FINE-TUNING ENABLED")
print("="*60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

# === SAVE FINAL MODEL ===
model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"📋 Classes: {class_names}")

# === SAVE TRAINING HISTORY ===
with open(HISTORY_PATH, 'w') as f:
    json.dump(history.history, f, indent=2)
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

# === PER-CLASS ACCURACY ===
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

val_predictions = model.predict(val_ds)
val_pred_classes = np.argmax(val_predictions, axis=1)

# Get true labels
val_true_labels = []
for _, label in val_items:
    val_true_labels.append(label)
val_true_labels = np.array(val_true_labels)

# Calculate per-class accuracy
for idx, class_name in enumerate(class_names):
    class_mask = val_true_labels == idx
    if np.sum(class_mask) > 0:
        class_acc = np.mean(val_pred_classes[class_mask] == idx)
        print(f"   {class_name}: {class_acc:.4f} ({class_acc*100:.1f}%)")

# === CONFUSION ANALYSIS ===
from collections import defaultdict
confusion_counts = defaultdict(int)
for true_label, pred_label in zip(val_true_labels, val_pred_classes):
    if true_label != pred_label:
        confusion_counts[(class_names[true_label], class_names[pred_label])] += 1

if confusion_counts:
    print("\n🔍 Most Common Confusions (Top 5):")
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for (true_class, pred_class), count in sorted_confusions:
        print(f"   {true_class} → {pred_class}: {count} times")

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
print(f"Total Epochs Completed: {len(history.history['accuracy'])}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
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