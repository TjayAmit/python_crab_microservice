"""
Crab classifier training — CROPPED pipeline.

Key differences vs train_model_mobile_net_v2.py:
  1. Crops every image to its Label Studio bounding box BEFORE training,
     so the model sees the crab, not the background/tray. (The median box
     covers only ~23% of the image; the old script fed 100% of the photo,
     i.e. ~77% background, which let the model cheat off the per-session
     background instead of learning the crab.)
  2. Adds a real held-out TEST set (train/val/test) so the final number
     is honest and never used for tuning.
  3. Fixes the augmentation order/ranges (augment on [0,255] BEFORE
     mobilenet preprocess_input, not after — the old order distorted colors).
  4. Prints a confusion matrix + species-only and gender-only accuracy so
     you can see exactly which classes get swapped.

This script does not require any new labeling — it uses the bbox coords
already in coco/result.json.
"""

import os
import json
import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from sklearn.utils.class_weight import compute_class_weight

# === CONFIG ===
COCO_DIR = "coco"
IMAGES_DIR = os.path.join(COCO_DIR, "images")
ANNOTATIONS_FILE = os.path.join(COCO_DIR, "result.json")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "my_model_cropped.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history_cropped.json")
REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BOX_MARGIN = 0.10          # add 10% padding around the box before cropping
MIN_BOX_PX = 8             # skip degenerate boxes smaller than this
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === LOAD COCO ANNOTATIONS ===
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
class_names = [categories[i] for i in sorted(categories.keys())]
num_classes = len(class_names)
category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

print(f"\nFound {num_classes} classes:")
for idx, name in enumerate(class_names):
    print(f"   {idx}: {name}")

image_annotations = defaultdict(list)
for ann in coco_data['annotations']:
    image_annotations[ann['image_id']].append(ann)

image_info = {}
for img in coco_data['images']:
    file_name = img['file_name']
    for path in (os.path.join(COCO_DIR, file_name),
                 os.path.join(IMAGES_DIR, file_name),
                 file_name):
        if os.path.exists(path):
            image_info[img['id']] = path
            break

print(f"\nFound {len(image_info)}/{len(coco_data['images'])} valid image files")

# Build dataset items as (path, class_idx, bbox=[x,y,w,h]).
# One item per annotation so multi-crab images contribute every labeled crab.
class_counts = defaultdict(int)
dataset_items = []
skipped_small = 0

for img_id, annotations in image_annotations.items():
    if img_id not in image_info:
        continue
    img_path = image_info[img_id]
    for ann in annotations:
        x, y, w, h = ann['bbox']
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            skipped_small += 1
            continue
        class_idx = category_id_to_index[ann['category_id']]
        dataset_items.append((img_path, class_idx, [float(x), float(y), float(w), float(h)]))
        class_counts[class_idx] += 1

print(f"\nUsable annotations: {len(dataset_items)} (skipped {skipped_small} tiny boxes)")
print("\nClass distribution:")
for class_idx in range(num_classes):
    print(f"   {class_names[class_idx]}: {class_counts[class_idx]} boxes")

if len(dataset_items) < 100:
    print("\nERROR: Too few images! Need at least 100 total.")
    raise SystemExit(1)

# === STRATIFIED TRAIN / VAL / TEST SPLIT ===
train_items, val_items, test_items = [], [], []
for class_idx in range(num_classes):
    items = [it for it in dataset_items if it[1] == class_idx]
    np.random.shuffle(items)
    n = len(items)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VAL_SPLIT)
    test_items.extend(items[:n_test])
    val_items.extend(items[n_test:n_test + n_val])
    train_items.extend(items[n_test + n_val:])

np.random.shuffle(train_items)
np.random.shuffle(val_items)
np.random.shuffle(test_items)

print(f"\nSplit -> train: {len(train_items)}  val: {len(val_items)}  test: {len(test_items)}")

train_labels = [it[1] for it in train_items]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

os.makedirs(MODEL_DIR, exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

# === DATASET PIPELINE ===
print("\n" + "=" * 60)
print("BUILDING DATA PIPELINE (with bbox cropping)")
print("=" * 60)


def load_and_crop(img_path, bbox, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    shape = tf.shape(img)
    H = tf.cast(shape[0], tf.float32)
    W = tf.cast(shape[1], tf.float32)

    x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
    mx = bw * BOX_MARGIN
    my = bh * BOX_MARGIN

    x1 = tf.clip_by_value(x - mx, 0.0, W - 1.0)
    y1 = tf.clip_by_value(y - my, 0.0, H - 1.0)
    x2 = tf.clip_by_value(x + bw + mx, 1.0, W)
    y2 = tf.clip_by_value(y + bh + my, 1.0, H)

    x1i = tf.cast(x1, tf.int32)
    y1i = tf.cast(y1, tf.int32)
    cw = tf.maximum(tf.cast(x2 - x1, tf.int32), 1)
    ch = tf.maximum(tf.cast(y2 - y1, tf.int32), 1)

    img = tf.image.crop_to_bounding_box(img, y1i, x1i, ch, cw)
    img = tf.image.resize(img, IMAGE_SIZE, method='bilinear')   # float32, range ~[0,255]
    return img, label


def augment_image(img, label):
    # Operates on [0,255] BEFORE preprocess_input (correct order).
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 25.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.image.random_hue(img, 0.02)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def preprocess(img, label):
    img = applications.mobilenet_v2.preprocess_input(img)   # -> [-1, 1]
    return img, label


def create_dataset(items, training=False, augment=False):
    paths = [it[0] for it in items]
    labels = [it[1] for it in items]
    boxes = [it[2] for it in items]

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(boxes, dtype=tf.float32), tf.constant(labels))
    )
    ds = ds.map(load_and_crop, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=1000, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = create_dataset(train_items, training=True, augment=True)
val_ds = create_dataset(val_items)
test_ds = create_dataset(test_items)
print("Data pipeline ready")

# === BUILD MODEL ===
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

base_model = applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet',
    pooling=None,
)
base_model.trainable = False

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

model = keras.Model(inputs, outputs, name='crab_classifier_cropped')
print(f"Total params: {model.count_params():,}")

metrics = ['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')]


def make_callbacks(patience):
    return [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                      restore_best_weights=True, mode='max', verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "checkpoint_best_cropped.keras"),
                                        monitor='val_accuracy', save_best_only=True, mode='max', verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.TerminateOnNaN(),
    ]


# === PHASE 1: train head ===
print("\n" + "=" * 60)
print("PHASE 1: training classifier head (base frozen)")
print("=" * 60)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=metrics)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
               callbacks=make_callbacks(15), class_weight=class_weight_dict, verbose=1)
p1 = len(h1.history['accuracy'])
print(f"Phase 1 done: {p1} epochs")

# === PHASE 2: fine-tune top of backbone ===
print("\n" + "=" * 60)
print("PHASE 2: fine-tuning top layers")
print("=" * 60)
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy', metrics=metrics)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=p1 + 100, initial_epoch=p1,
               callbacks=make_callbacks(10), class_weight=class_weight_dict, verbose=1)

model.save(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")


def cat(a, b):
    return list(a) + list(b)


combined = {k: cat(h1.history.get(k, []), h2.history.get(k, []))
            for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss',
                      'top_2_accuracy', 'val_top_2_accuracy']}
with open(HISTORY_PATH, 'w') as f:
    json.dump(combined, f, indent=2)

# === EVALUATION ON HELD-OUT TEST SET ===
print("\n" + "=" * 60)
print("FINAL EVALUATION (held-out TEST set)")
print("=" * 60)

test_loss, test_acc, test_top2 = model.evaluate(test_ds, verbose=0)
train_eval = model.evaluate(create_dataset(train_items), verbose=0)
print(f"\nTRAIN acc: {train_eval[1]:.4f}")
print(f"TEST  acc: {test_acc:.4f}   top-2: {test_top2:.4f}   loss: {test_loss:.4f}")
print(f"Train-Test gap: {train_eval[1] - test_acc:.4f}")

# Predictions on the test set for confusion analysis
y_true = np.array([it[1] for it in test_items])
y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Confusion matrix
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

print("\nCONFUSION MATRIX (rows = true, cols = predicted)")
header = "true \\ pred".ljust(22) + " ".join(f"{i:>4}" for i in range(num_classes))
print(header)
for i in range(num_classes):
    row = " ".join(f"{cm[i, j]:>4}" for j in range(num_classes))
    print(f"{i}:{class_names[i][:18]:<20} {row}")

print("\nPer-class precision / recall:")
for i in range(num_classes):
    tp = cm[i, i]
    recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0
    precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0
    print(f"  {class_names[i]:<20} precision={precision:.2%}  recall={recall:.2%}")


# === SPECIES-ONLY and GENDER-ONLY accuracy ===
def split_species_gender(name):
    for g in ('Female', 'Male', 'Gay'):
        if name.endswith(g):
            return name[: -len(g)].strip(), g
    return name, 'N/A'


true_sp = [split_species_gender(class_names[t])[0] for t in y_true]
pred_sp = [split_species_gender(class_names[p])[0] for p in y_pred]
species_acc = np.mean([a == b for a, b in zip(true_sp, pred_sp)])

true_g = [split_species_gender(class_names[t])[1] for t in y_true]
pred_g = [split_species_gender(class_names[p])[1] for p in y_pred]
gender_mask = [g != 'N/A' for g in true_g]
if any(gender_mask):
    gender_acc = np.mean([tg == pg for tg, pg, m in zip(true_g, pred_g, gender_mask) if m])
else:
    gender_acc = float('nan')

print("\n" + "=" * 60)
print("DECOMPOSED ACCURACY (the numbers that matter for you)")
print("=" * 60)
print(f"  SPECIES-only accuracy: {species_acc:.2%}   (Blue / Mud / Curacha / Killer)")
print(f"  GENDER-only accuracy : {gender_acc:.2%}   (only over crabs that have a sex label)")
print(f"  FULL 9-class accuracy: {test_acc:.2%}")

report = {
    'test_accuracy': float(test_acc),
    'test_top2': float(test_top2),
    'train_accuracy': float(train_eval[1]),
    'species_accuracy': float(species_acc),
    'gender_accuracy': float(gender_acc),
    'confusion_matrix': cm.tolist(),
    'class_names': class_names,
}
with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nSaved evaluation report -> {REPORT_PATH}")
print("\nTRAINING COMPLETE")
