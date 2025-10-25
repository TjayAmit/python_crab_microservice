import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import os
import json

# === CONFIG ===
DATASET_DIR = "curacha_dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 16  # Reduced for better gradient updates with small datasets
EPOCHS = 50  # Increased for better learning

# === LOAD DATA ===
data_dir = pathlib.Path(DATASET_DIR)

# Load datasets WITHOUT augmentation for validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names automatically
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Classes found: {class_names}")
print(f"📊 Number of classes: {num_classes}")

# Save class names for FastAPI later
os.makedirs(MODEL_DIR, exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)
print(f"📁 Saved class names to {CLASS_NAMES_PATH}")

# === DATA AUGMENTATION ===
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),  # 30% rotation
    layers.RandomZoom(0.2),  # 20% zoom
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# === PERFORMANCE OPTIMIZATION ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === IMPROVED MODEL ARCHITECTURE ===
# Using transfer learning base + custom head for better feature extraction
base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model initially

model = keras.Sequential([
    # Preprocessing
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    
    # Data augmentation (only during training)
    data_augmentation,
    
    # Transfer learning base
    base_model,
    
    # Custom classification head
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

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Freeze first 100 layers, fine-tune the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  # Additional epochs for fine-tuning
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# === SAVE FINAL MODEL ===
model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"📋 Classes: {class_names}")

# === SAVE TRAINING HISTORY ===
# Combine both training phases
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

# === EVALUATE ON TRAINING (NO SHUFFLE) ===
train_eval_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False
)

train_loss, train_acc = model.evaluate(train_eval_ds)
print(f"✅ Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"✅ Training Loss: {train_loss:.4f}")

# === INTERPRET RESULTS ===
print("\n" + "="*60)
print("TRAINING ANALYSIS")
print("="*60)

# Calculate overfitting indicator
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