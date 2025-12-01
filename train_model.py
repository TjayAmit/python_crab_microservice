import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import pathlib
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === GPU CONFIGURATION ===
print("\n" + "="*60)
print("🖥️  GPU CONFIGURATION")
print("="*60)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TF from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Set GPU as visible device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ Using: {logical_gpus[0].name}")
        
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
else:
    print("⚠️  No GPU found! Training will use CPU (much slower)")
    print("   Make sure CUDA and cuDNN are installed properly")

# === ENABLE MIXED PRECISION FOR GTX 1080 ===
# GTX 1080 benefits from mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f'\n🚀 Mixed precision enabled: {policy.name}')
print(f'   Compute dtype: {policy.compute_dtype}')
print(f'   Variable dtype: {policy.variable_dtype}')
print(f'   Optimized for GTX 1080 performance')

# === CONFIG ===
COCO_DIR = "coco"
IMAGES_DIR = os.path.join(COCO_DIR, "images")
ANNOTATIONS_FILE = os.path.join(COCO_DIR, "result.json")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "my_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

# Critical hyperparameters optimized for GTX 1080 (8GB VRAM)
IMAGE_SIZE = (224, 224)  # Balanced resolution for GTX 1080
BATCH_SIZE = 32  # Optimized for GTX 1080 memory
EPOCHS = 150  # More epochs for convergence
VALIDATION_SPLIT = 0.2
AUTO_LR_FINDER = True  # Enable automatic learning rate finding

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# === LOAD COCO ANNOTATIONS ===
print("\n" + "="*60)
print("📁 LOADING DATASET")
print("="*60)

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
    
    if '..' in file_name or file_name.startswith('label-studio'):
        file_name = os.path.basename(file_name)
    
    possible_paths = [
        os.path.join(IMAGES_DIR, file_name),
        os.path.join(COCO_DIR, file_name),
        file_name,
    ]
    
    original_path = os.path.join(COCO_DIR, img['file_name'])
    if os.path.exists(original_path):
        possible_paths.insert(0, original_path)
    
    for path in possible_paths:
        if os.path.exists(path):
            image_info[img_id] = path
            break

print(f"📊 Found {len(image_info)} valid images")

# Prepare dataset with class distribution
dataset_items = []
class_counts = {i: 0 for i in range(num_classes)}

for img_id, annotations in image_annotations.items():
    if img_id in image_info:
        img_path = image_info[img_id]
        if os.path.exists(img_path):
            category_id = annotations[0]['category_id']
            class_idx = category_id_to_index[category_id]
            dataset_items.append((img_path, class_idx))
            class_counts[class_idx] += 1

print(f"\n📊 Dataset Distribution:")
for idx, name in enumerate(class_names):
    print(f"   {name}: {class_counts[idx]} images")

total_images = len(dataset_items)
print(f"\n✅ Total images: {total_images}")

# Calculate class weights for imbalanced data
class_weights = {}
max_count = max(class_counts.values())
for idx in range(num_classes):
    class_weights[idx] = max_count / (class_counts[idx] + 1e-6)
print(f"\n⚖️  Class weights calculated for balanced training")

# Stratified split to maintain class distribution
from collections import defaultdict
class_items = defaultdict(list)
for item in dataset_items:
    class_items[item[1]].append(item)

train_items = []
val_items = []
np.random.seed(123)

for class_idx, items in class_items.items():
    np.random.shuffle(items)
    split_idx = int(len(items) * (1 - VALIDATION_SPLIT))
    train_items.extend(items[:split_idx])
    val_items.extend(items[split_idx:])

np.random.shuffle(train_items)
np.random.shuffle(val_items)

print(f"\n📊 Training samples: {len(train_items)}")
print(f"📊 Validation samples: {len(val_items)}")

# Save class names
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# === ADVANCED DATA AUGMENTATION FOR CRAB CLASSIFICATION ===
print("\n" + "="*60)
print("🎨 CONFIGURING DATA AUGMENTATION")
print("="*60)

# Optimized augmentation for crab gender/species classification
# Focus on variations that preserve critical gender characteristics
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Crabs can be mirrored
    layers.RandomRotation(0.15),  # Moderate rotation (crabs have orientation)
    layers.RandomZoom(0.15),  # Moderate zoom to focus on details
    layers.RandomTranslation(0.1, 0.1),  # Slight translation
    layers.RandomContrast(0.2),  # Contrast variation for different lighting
    layers.RandomBrightness(0.2),  # Brightness variation
], name='augmentation')

print("✅ Augmentation pipeline configured:")
print("   - Horizontal flip for symmetry")
print("   - Moderate rotation (15%) to preserve orientation")
print("   - Controlled zoom (15%) for detail focus")
print("   - Brightness/contrast for lighting robustness")

# === ADVANCED DATA PIPELINE ===
def load_and_preprocess_image(img_path, label):
    """Load and preprocess with advanced techniques"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    
    # Resize with proper aspect ratio handling
    img = tf.image.resize(img, IMAGE_SIZE, method='bicubic')
    
    return img, label

def create_dataset(items, is_training=True):
    """Create optimized TensorFlow dataset with GPU-friendly settings"""
    paths = [item[0] for item in items]
    labels = [item[1] for item in items]
    
    # Use AUTOTUNE for optimal GPU utilization
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # Parallel data loading for GPU
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE,
                         deterministic=False)
    
    if is_training:
        dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()  # Infinite dataset for steps_per_epoch
    
    # Batch and prefetch for GPU performance
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

steps_per_epoch = len(train_items) // BATCH_SIZE
validation_steps = len(val_items) // BATCH_SIZE

train_ds = create_dataset(train_items, is_training=True)
val_ds = create_dataset(val_items, is_training=False)

print(f"\n✅ Data pipeline optimized for GPU:")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Validation steps: {validation_steps}")
print(f"   Batch size: {BATCH_SIZE} (optimized for GTX 1080)")
print(f"   Prefetching: AUTOTUNE (parallel GPU loading)")

# === LEARNING RATE FINDER (FAST VERSION) ===
if AUTO_LR_FINDER:
    print("\n" + "="*60)
    print("🔍 AUTOMATIC LEARNING RATE FINDER (FAST)")
    print("="*60)
    
    # Create a lightweight dataset for LR finding (no shuffle, smaller)
    lr_find_steps = min(100, steps_per_epoch)  # Only 100 steps needed
    lr_find_ds = tf.data.Dataset.from_tensor_slices(([item[0] for item in train_items[:BATCH_SIZE*lr_find_steps]], 
                                                       [item[1] for item in train_items[:BATCH_SIZE*lr_find_steps]]))
    lr_find_ds = lr_find_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    lr_find_ds = lr_find_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Create a simple model for LR finding
    lr_model = keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Rescaling(1./255),
        keras.applications.EfficientNetB0(include_top=False, weights='imagenet'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    # LR range test
    min_lr = 1e-7
    max_lr = 1e-2
    
    class LRFinder(keras.callbacks.Callback):
        def __init__(self, min_lr, max_lr, total_steps):
            super().__init__()
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.total_steps = total_steps
            self.lrs = []
            self.losses = []
            self.iteration = 0
            
        def on_batch_end(self, batch, logs=None):
            lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.iteration / self.total_steps)
            self.model.optimizer.learning_rate = lr
            self.lrs.append(lr)
            self.losses.append(logs['loss'])
            self.iteration += 1
            
            # Stop if loss explodes
            if self.iteration > 10 and logs['loss'] > self.losses[0] * 4:
                self.model.stop_training = True
    
    lr_finder = LRFinder(min_lr, max_lr, lr_find_steps)
    
    lr_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=min_lr),
        loss='sparse_categorical_crossentropy'
    )
    
    print(f"🔄 Running LR finder on {lr_find_steps} steps (fast mode)...")
    with tf.device('/GPU:0'):
        lr_model.fit(
            lr_find_ds,
            steps_per_epoch=lr_find_steps,
            epochs=1,
            callbacks=[lr_finder],
            verbose=0
        )
    
    # Find optimal LR (steepest descent)
    losses = np.array(lr_finder.losses)
    lrs = np.array(lr_finder.lrs)
    
    if len(losses) > 20:
        # Smooth losses
        window = min(20, len(losses) // 5)
        smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        smoothed_lrs = lrs[window-1:len(smoothed_losses) + window - 1]
        
        # Find steepest gradient
        gradients = np.gradient(smoothed_losses)
        optimal_idx = np.argmin(gradients)
        optimal_lr = smoothed_lrs[optimal_idx]
        
        # Use 1/10th of optimal for safety
        initial_learning_rate = optimal_lr / 10
        
        print(f"✅ Optimal learning rate found: {initial_learning_rate:.2e}")
        
        # Plot LR finder results
        plt.figure(figsize=(10, 6))
        plt.semilogx(smoothed_lrs, smoothed_losses)
        plt.axvline(optimal_lr, color='r', linestyle='--', label=f'Optimal: {optimal_lr:.2e}')
        plt.axvline(initial_learning_rate, color='g', linestyle='--', label=f'Selected: {initial_learning_rate:.2e}')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, 'lr_finder.png'), dpi=150, bbox_inches='tight')
        print(f"📊 LR finder plot saved to {PLOTS_DIR}/lr_finder.png")
        plt.close()
    else:
        print("⚠️  Not enough data points, using default LR")
        initial_learning_rate = 1e-4
    
    del lr_model, lr_find_ds  # Free memory
    import gc
    gc.collect()
else:
    initial_learning_rate = 1e-4
    print(f"\n📊 Using default learning rate: {initial_learning_rate:.2e}")

# === BUILD OPTIMIZED MODEL ===
print("\n" + "="*60)
print("🏗️  BUILDING MODEL ARCHITECTURE")
print("="*60)

# EfficientNetB0 with optimized fine-tuning strategy
base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Strategic layer unfreezing for crab classification
# Freeze batch normalization layers to maintain pretrained statistics
base_model.trainable = True
fine_tune_at = 100  # Unfreeze from layer 100 onwards

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
# Keep all batch norm layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"🔓 Base model: {trainable_count}/{len(base_model.layers)} layers trainable")

# Advanced model head for gender classification
inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)

# Base model
x = base_model(x, training=False)  # Use BN in inference mode

# Custom classification head optimized for subtle differences
x = layers.GlobalAveragePooling2D(name='gap')(x)

# First dense block
x = layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)

# Second dense block  
x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.4)(x)

# Third dense block
x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)

# Output layer (float32 for numerical stability)
outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='crab_gender_classifier')

print(f"\n✅ Model architecture:")
print(f"   Base: EfficientNetB0 (ImageNet pretrained)")
print(f"   Head: 1024 → 512 → 256 → {num_classes}")
print(f"   Total parameters: {model.count_params():,}")
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"   Trainable parameters: {trainable_params:,}")

# === COMPILE WITH ADVANCED OPTIMIZER ===
# Cosine decay with warmup
total_steps = steps_per_epoch * EPOCHS
warmup_steps = steps_per_epoch * 10

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=total_steps - warmup_steps,
    alpha=0.0001  # Final LR will be initial_lr * alpha
)

# Warmup schedule
class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, initial_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.cosine_decay = keras.optimizers.schedules.CosineDecay(
            initial_lr, total_steps - warmup_steps, alpha=0.0001
        )
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        else:
            return self.cosine_decay(step - self.warmup_steps)

lr_schedule = WarmUpCosineDecay(warmup_steps, total_steps, initial_learning_rate)

# AdamW optimizer (Adam with weight decay)
optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✅ Optimizer configured:")
print(f"   Type: Adam with cosine decay + warmup")
print(f"   Initial LR: {initial_learning_rate:.2e}")
print(f"   Warmup steps: {warmup_steps}")
print(f"   Total steps: {total_steps}")

# === ADVANCED CALLBACKS ===
print("\n" + "="*60)
print("⚙️  CONFIGURING TRAINING CALLBACKS")
print("="*60)

callbacks_list = []

# Early stopping with restore best weights
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    mode='max',
    restore_best_weights=True,
    verbose=1
)
callbacks_list.append(early_stop)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.keras"),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)
callbacks_list.append(checkpoint)

# CSV logger
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(MODEL_DIR, 'training_log.csv'),
    append=False
)
callbacks_list.append(csv_logger)

# Custom callback for detailed logging
class DetailedLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, '__call__'):
            lr = lr(self.model.optimizer.iterations).numpy()
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   LR: {lr:.2e} | Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.4f}")
        print(f"   Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}")

callbacks_list.append(DetailedLogger())

print("✅ Callbacks configured:")
print("   - Early stopping (patience=20)")
print("   - Model checkpoint (best val_accuracy)")
print("   - CSV logging")
print("   - Detailed epoch logging")

# === TRAINING ===
print("\n" + "="*60)
print("🚀 STARTING TRAINING ON GPU")
print("="*60)
print(f"GPU: GTX 1080 (Mixed Precision Enabled)")
print(f"Target: Classify 3 crab species × 3 genders = {num_classes} classes")
print(f"Training for up to {EPOCHS} epochs with early stopping")
print(f"Batch size: {BATCH_SIZE} | Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
print("="*60 + "\n")

# Monitor GPU usage during training
with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )

# === SAVE MODEL ===
print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

with open(HISTORY_PATH, 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    json.dump(history_dict, f, indent=2)
print(f"✅ History saved to {HISTORY_PATH}")

# === COMPREHENSIVE EVALUATION ===
print("\n" + "="*60)
print("📊 COMPREHENSIVE EVALUATION")
print("="*60)

# Evaluate on validation set
val_eval_ds = create_dataset(val_items, is_training=False)
val_loss, val_acc = model.evaluate(val_eval_ds, steps=validation_steps, verbose=0)

# Evaluate on training set
train_eval_ds = create_dataset(train_items, is_training=False)
train_loss, train_acc = model.evaluate(train_eval_ds, steps=steps_per_epoch, verbose=0)

print(f"\n✅ Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"✅ Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"✅ Training Loss: {train_loss:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")

# === DETAILED PER-CLASS ANALYSIS ===
print("\n" + "="*60)
print("🔍 PER-CLASS PERFORMANCE ANALYSIS")
print("="*60)

# Get predictions
val_predictions = model.predict(val_eval_ds, steps=validation_steps, verbose=0)
val_pred_classes = np.argmax(val_predictions, axis=1)

# Get true labels
val_true_labels = np.array([label for _, label in val_items])[:len(val_pred_classes)]

# Classification report
print("\n📋 Classification Report:")
print(classification_report(val_true_labels, val_pred_classes, 
                          target_names=class_names, digits=4))

# Per-class accuracy
print("\n📊 Per-Class Accuracy:")
for idx, class_name in enumerate(class_names):
    class_mask = val_true_labels == idx
    if np.sum(class_mask) > 0:
        class_acc = np.mean(val_pred_classes[class_mask] == idx)
        class_samples = np.sum(class_mask)
        print(f"   {class_name:25s}: {class_acc:.4f} ({class_acc*100:.1f}%) - {class_samples} samples")

# === CONFUSION MATRIX ===
print("\n" + "="*60)
print("🔍 CONFUSION MATRIX ANALYSIS")
print("="*60)

cm = confusion_matrix(val_true_labels, val_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Crab Gender Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix saved to {PLOTS_DIR}/confusion_matrix.png")
plt.close()

# Most confused pairs
print("\n🔍 Most Common Misclassifications:")
from collections import defaultdict
confusion_pairs = defaultdict(int)
for true_idx, pred_idx in zip(val_true_labels, val_pred_classes):
    if true_idx != pred_idx:
        confusion_pairs[(class_names[true_idx], class_names[pred_idx])] += 1

sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
for i, ((true_class, pred_class), count) in enumerate(sorted_confusions, 1):
    print(f"   {i}. {true_class} → {pred_class}: {count} times")

# === TRAINING CURVES ===
print("\n" + "="*60)
print("📈 GENERATING TRAINING CURVES")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Per-class accuracy bar chart
class_accuracies = []
for idx in range(num_classes):
    class_mask = val_true_labels == idx
    if np.sum(class_mask) > 0:
        class_acc = np.mean(val_pred_classes[class_mask] == idx)
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

axes[1, 0].bar(range(num_classes), class_accuracies, color='steelblue')
axes[1, 0].set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_xticks(range(num_classes))
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].axhline(y=val_acc, color='r', linestyle='--', label=f'Overall: {val_acc:.3f}')
axes[1, 0].legend()

# Class distribution
train_class_counts = [class_counts[i] for i in range(num_classes)]
axes[1, 1].bar(range(num_classes), train_class_counts, color='coral')
axes[1, 1].set_title('Dataset Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Number of Images')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_xticks(range(num_classes))
axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_analysis.png'), dpi=150, bbox_inches='tight')
print(f"✅ Training curves saved to {PLOTS_DIR}/training_analysis.png")
plt.close()

# === FINAL SUMMARY ===
print("\n" + "="*60)
print("📋 TRAINING SUMMARY")
print("="*60)

overfitting_gap = train_acc - val_acc

print(f"\n🎯 Final Results:")
print(f"   Training Accuracy:    {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Validation Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Generalization Gap:   {overfitting_gap:.4f}")

print(f"\n📊 Training Details:")
print(f"   Total Epochs:         {len(history.history['accuracy'])}")
print(f"   Best Val Accuracy:    {max(history.history['val_accuracy']):.4f}")
print(f"   Images Processed:     {total_images:,}")
print(f"   Classes:              {num_classes}")

print(f"\n💾 Saved Artifacts:")
print(f"   Model:                {MODEL_PATH}")
print(f"   Class Names:          {CLASS_NAMES_PATH}")
print(f"   Training History:     {HISTORY_PATH}")
print(f"   Plots:                {PLOTS_DIR}/")

if val_acc >= 0.90:
    print(f"\n🎉 EXCELLENT! Model achieved {val_acc*100:.1f}% accuracy!")
    print("   Ready for production deployment.")
elif val_acc >= 0.80:
    print(f"\n✅ GOOD! Model achieved {val_acc*100:.1f}% accuracy.")
    print("   Consider collecting more data for challenging classes.")
elif val_acc >= 0.70:
    print(f"\n🔄 DECENT. Model achieved {val_acc*100:.1f}% accuracy.")
    print("   Review confused classes and collect targeted data.")
else:
    print(f"\n⚠️  Model achieved {val_acc*100:.1f}% accuracy.")
    print("   Review data quality and consider more training.")

print("\n" + "="*60)
print("✨ TRAINING COMPLETE!")
print("="*60)
print("\n💡 Next Steps:")
print("   1. Review plots in model/plots/ directory")
print("   2. Check confusion matrix for problem areas")
print("   3. Run: uvicorn main:app --reload")
print("   4. Test: http://localhost:8000/test_accuracy")
print("="*60 + "\n")