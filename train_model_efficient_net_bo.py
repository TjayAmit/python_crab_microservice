import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight

# === CONFIG ===
COCO_DIR = "coco"
IMAGES_DIR = os.path.join(COCO_DIR, "images")
ANNOTATIONS_FILE = os.path.join(COCO_DIR, "result.json")
MODEL_DIR = "model_efficientnet"
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === LOAD COCO ANNOTATIONS ===
print("="*60)
print("📁 LOADING DATA")
print("="*60)

with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
class_names = [categories[i] for i in sorted(categories.keys())]
num_classes = len(class_names)
category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

print(f"\n✅ Found {num_classes} classes:")
for idx, name in enumerate(class_names):
    print(f"   {idx}: {name}")

image_annotations = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

image_info = {}
for img in coco_data['images']:
    img_id = img['id']
    file_name = img['file_name']

    possible_paths = [
        os.path.join(COCO_DIR, file_name),
        os.path.join(IMAGES_DIR, file_name),
        file_name,
    ]

    for path in possible_paths:
        if os.path.exists(path):
            image_info[img_id] = path
            break

print(f"\n📊 Found {len(image_info)}/{len(coco_data['images'])} valid image files")

class_counts = defaultdict(int)
dataset_items = []

for img_id, annotations in image_annotations.items():
    if img_id in image_info:
        img_path = image_info[img_id]
        if os.path.exists(img_path):
            category_id = annotations[0]['category_id']
            class_idx = category_id_to_index[category_id]
            dataset_items.append((img_path, class_idx))
            class_counts[class_idx] += 1

print(f"\n📊 Class distribution:")
for class_idx in range(num_classes):
    count = class_counts[class_idx]
    print(f"   {class_names[class_idx]}: {count} images")

if len(dataset_items) < 100:
    print("\n❌ ERROR: Too few images! Need at least 100 images total.")
    exit(1)

train_items = []
val_items = []

for class_idx in range(num_classes):
    class_items = [item for item in dataset_items if item[1] == class_idx]
    np.random.shuffle(class_items)

    split_idx = int(len(class_items) * (1 - VALIDATION_SPLIT))
    train_items.extend(class_items[:split_idx])
    val_items.extend(class_items[split_idx:])

np.random.shuffle(train_items)
np.random.shuffle(val_items)

print(f"\n✅ Split complete:")
print(f"   Training: {len(train_items)} images")
print(f"   Validation: {len(val_items)} images")

train_labels = [item[1] for item in train_items]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"\n⚖️ Class weights computed (to handle imbalance)")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

# === DATASET PIPELINE ===
print("\n" + "="*60)
print("🔧 BUILDING DATA PIPELINE")
print("="*60)

def load_and_preprocess_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE, method='bilinear')
    img = tf.cast(img, tf.float32)
    return img, label

def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2 * 255)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label

def create_dataset(items, is_training=True, augment=False):
    paths = [item[0] for item in items]
    labels = [item[1] for item in items]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(train_items, is_training=True, augment=True)
val_ds = create_dataset(val_items, is_training=False, augment=False)

print("✅ Data pipeline ready")

# === BUILD MODEL ===
print("\n" + "="*60)
print("🏗️ BUILDING MODEL")
print("="*60)

base_model = applications.EfficientNetB0(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet',
    pooling=None
)
base_model.trainable = False

print(f"✅ Loaded EfficientNetB0 (ImageNet weights)")
print(f"   Total layers: {len(base_model.layers)}")

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name='global_pool')(x)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Dropout(0.3, name='dropout1')(x)
x = layers.Dense(256, activation='relu', name='dense1')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.Dropout(0.2, name='dropout2')(x)
x = layers.Dense(128, activation='relu', name='dense2')(x)
x = layers.Dropout(0.1, name='dropout3')(x)
outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

model = keras.Model(inputs, outputs, name='efficientnet_classifier')

print(f"\n📊 Model architecture:")
print(f"   Input: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3")
print(f"   Base: EfficientNetB0 (frozen)")
print(f"   Head: GlobalPool → Dense(256) → Dense(128) → Dense({num_classes})")
print(f"   Total params: {model.count_params():,}")

initial_lr = 0.001

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
)

print(f"\n✅ Model compiled (learning_rate={initial_lr})")

# === PHASE 1 CALLBACKS ===
callbacks_phase1 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "checkpoint_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.TerminateOnNaN()
]

# === PHASE 1: TRAIN CLASSIFIER HEAD ===
print("\n" + "="*60)
print("📚 PHASE 1: Training classifier head (base frozen)")
print("="*60)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict,
    verbose=1
)

phase1_epochs = len(history_phase1.history['accuracy'])
print(f"\n✅ Phase 1 complete: {phase1_epochs} epochs")

# === PHASE 2: FINE-TUNE ===
print("\n" + "="*60)
print("🔥 PHASE 2: Fine-tuning (unfreezing top layers)")
print("="*60)

base_model.trainable = True

fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"✅ Unfroze top {trainable_layers} layers of base model")

fine_tune_lr = 1e-5

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
)

print(f"✅ Recompiled (learning_rate={fine_tune_lr})")

# === PHASE 2 CALLBACKS ===
# FIX: Always create FRESH callbacks for Phase 2.
# Reusing Phase 1 callbacks causes EarlyStopping to trigger
# immediately because its internal patience counter is still
# exhausted from Phase 1, resulting in an empty history dict {}.
callbacks_phase2 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "checkpoint_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.TerminateOnNaN()
]

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=phase1_epochs + 50,   # FIX: ensure Phase 2 actually has epochs to run
    initial_epoch=phase1_epochs,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict,
    verbose=1
)

print("\n🔍 Phase 2 history keys:", list(history_phase2.history.keys()))
print("🔍 Phase 2 epochs ran:", len(history_phase2.history.get('accuracy', [])))

if 'accuracy' in history_phase2.history and len(history_phase2.history['accuracy']) > 0:
    phase2_epochs = len(history_phase2.history['accuracy'])
    print(f"\n✅ Phase 2 complete: {phase2_epochs} epochs")
else:
    print("\n⚠️ Phase 2 training did not complete or had no epochs")
    print("   Using phase 1 results only")
    phase2_epochs = 0
    history_phase2.history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': [],
        'top_2_accuracy': [],
        'val_top_2_accuracy': []
    }

# === SAVE MODEL ===
model.save(MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")

# === COMBINE HISTORY ===
combined_history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'top_2_accuracy': history_phase1.history['top_2_accuracy'] + history_phase2.history['top_2_accuracy'],
    'val_top_2_accuracy': history_phase1.history['val_top_2_accuracy'] + history_phase2.history['val_top_2_accuracy'],
}

with open(HISTORY_PATH, 'w') as f:
    json.dump(combined_history, f, indent=2)

# === FINAL EVALUATION ===
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

val_results = model.evaluate(val_ds, verbose=0)
val_loss, val_acc, val_top2 = val_results

train_eval_ds = create_dataset(train_items, is_training=False, augment=False)
train_results = model.evaluate(train_eval_ds, verbose=0)
train_loss, train_acc, train_top2 = train_results

print(f"\n📈 TRAINING SET:")
print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Top-2 Acc: {train_top2:.4f} ({train_top2*100:.2f}%)")
print(f"   Loss: {train_loss:.4f}")

print(f"\n📉 VALIDATION SET:")
print(f"   Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Top-2 Acc: {val_top2:.4f} ({val_top2*100:.2f}%)")
print(f"   Loss: {val_loss:.4f}")

overfitting_gap = train_acc - val_acc
print(f"\n📊 Overfitting gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")

# === PER-CLASS ANALYSIS ===
print("\n" + "="*60)
print("🔍 PER-CLASS PERFORMANCE")
print("="*60)

val_predictions = model.predict(val_ds, verbose=0)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = np.array([label for _, label in val_items])

print(f"\n{'Class':<20} {'Accuracy':<10} {'Correct/Total':<15}")
print("-" * 50)

class_accuracies = []
for class_idx in range(num_classes):
    class_mask = val_true_classes == class_idx
    class_total = np.sum(class_mask)

    if class_total > 0:
        class_correct = np.sum(val_pred_classes[class_mask] == class_idx)
        class_acc = class_correct / class_total
        class_accuracies.append(class_acc)
        print(f"{class_names[class_idx]:<20} {class_acc:>6.2%}     {class_correct:>3}/{class_total:<3}")
    else:
        print(f"{class_names[class_idx]:<20} {'N/A':<10} {'0/0':<15}")

avg_class_acc = np.mean(class_accuracies) if class_accuracies else 0
print("-" * 50)
print(f"{'Average':<20} {avg_class_acc:>6.2%}")

# === DIAGNOSIS ===
print("\n" + "="*60)
print("🔬 DIAGNOSIS")
print("="*60)

if val_acc < 0.30:
    print("\n❌ CRITICAL ISSUE: Model is performing poorly!")
    print("\n🔍 Possible causes:")
    print("   1. Images might be incorrectly labeled")
    print("   2. Images might be too similar between classes")
    print("   3. Image quality might be too poor")
    print("   4. Dataset might be too small")
    print("\n💡 Next steps:")
    print("   • Manually inspect images and labels")
    print("   • Check if humans can distinguish the classes")
    print("   • Verify images are loading correctly")
    print("   • Consider collecting more diverse images")
elif val_acc < 0.60:
    print("\n⚠️ Model is struggling to learn")
    print("\n💡 Suggestions:")
    print("   • Check image quality and labeling")
    print("   • Increase dataset size if possible")
    print("   • Verify classes are visually distinguishable")
elif val_acc < 0.80:
    print("\n🟡 Model is learning but could improve")
    if overfitting_gap > 0.15:
        print("   • Showing signs of overfitting")
        print("   • Consider more data augmentation")
    else:
        print("   • Consider training longer")
        print("   • Try different augmentation strategies")
else:
    print("\n✅ Model is performing well!")
    if overfitting_gap < 0.10:
        print("   • Excellent generalization!")
    else:
        print("   • Some overfitting detected, but acceptable")

# === SUMMARY ===
print("\n" + "="*60)
print("📋 TRAINING SUMMARY")
print("="*60)
print(f"Model: EfficientNetB0")
print(f"Total epochs: {len(combined_history['accuracy'])}")
print(f"Best val accuracy: {max(combined_history['val_accuracy']):.4f}")
print(f"Final train acc: {train_acc:.4f}")
print(f"Final val acc: {val_acc:.4f}")
print(f"\nFiles saved:")
print(f"  • {MODEL_PATH}")
print(f"  • {CLASS_NAMES_PATH}")
print(f"  • {HISTORY_PATH}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)