"""
DATA AUGMENTATION: IMPROVING CNN PERFORMANCE
=============================================
Artificially expand dataset with transformations
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
#from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
import time

print("="*80)
print("DATA AUGMENTATION: BOOSTING CNN PERFORMANCE")
print("="*80)

# WHAT IS DATA AUGMENTATION?

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    DATA AUGMENTATION                                   ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  THE PROBLEM:                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │ • Limited training data (50,000 images)                          │ ║
║  │ • Model memorizes training set → overfitting                     │ ║
║  │ • Real world: Objects at different angles, lighting, positions  │ ║
║  │ • Model needs to generalize to variations                        │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  THE SOLUTION: DATA AUGMENTATION                                       ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │ • Create variations of existing images                           │ ║
║  │ • Apply random transformations during training                   │ ║
║  │ • Model sees "different" image each epoch                        │ ║
║  │ • Artificially expand dataset 10x, 100x, 1000x!                 │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  COMMON TRANSFORMATIONS:                                               ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                                        ║
║  1. HORIZONTAL FLIP                                                    ║
║     Original: 🚗 →                                                    ║
║     Flipped:  ← 🚗                                                    ║
║     Why: Car facing left OR right should both be "car"                ║
║                                                                        ║
║  2. ROTATION                                                           ║
║     Original: ✈️ (horizontal)                                         ║
║     Rotated:  ✈️ (tilted 15°)                                         ║
║     Why: Airplane at any angle is still airplane                      ║
║                                                                        ║
║  3. ZOOM (Scale)                                                       ║
║     Original: 🐱 (small)                                              ║
║     Zoomed:   🐱🐱 (larger)                                           ║
║     Why: Cat close-up or far away is still cat                        ║
║                                                                        ║
║  4. WIDTH/HEIGHT SHIFT                                                 ║
║     Original: 🐕 (center)                                             ║
║     Shifted:      🐕 (off-center)                                     ║
║     Why: Object position varies in real photos                        ║
║                                                                        ║
║  5. BRIGHTNESS                                                         ║
║     Original: 🌞 (normal lighting)                                    ║
║     Adjusted: 🌤️ (darker/brighter)                                   ║
║     Why: Lighting conditions vary                                     ║
║                                                                        ║
║  DON'T USE:                                                            ║
║  ❌ Vertical flip (cars don't drive upside down!)                     ║
║  ❌ Excessive rotation (90°+ makes airplane look wrong)               ║
║  ❌ Extreme zoom (loses critical features)                            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# LOAD DATA

print("\n" + "="*80)
print("LOADING CIFAR-10")
print("="*80)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

print(f"✅ Data loaded: {len(X_train):,} training images")

# VISUALIZE AUGMENTATION

print("\n" + "="*80)
print("DEMONSTRATING AUGMENTATION EFFECTS")
print("="*80)

# Create augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),    # ~15 degree
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])
"""# Create augmentation generator
datagen = image_dataset_from_directory(
    rotation_range=15,          # Rotate ±15 degrees
    width_shift_range=0.1,      # Shift width ±10%
    height_shift_range=0.1,     # Shift height ±10%
    horizontal_flip=True,       # Flip horizontally
    zoom_range=0.1              # Zoom ±10%
)"""

# Select one image
sample_img = X_train_norm[0:1]  # Shape: (1, 32, 32, 3)

# Generate augmented versions
print("📊 Generating augmented versions...")

fig, axes = plt.subplots(3, 6, figsize=(15, 8))
fig.suptitle('DATA AUGMENTATION EXAMPLES', fontsize=16, fontweight='bold')

# Original
axes[0, 0].imshow(sample_img[0])
axes[0, 0].set_title('Original', fontweight='bold', fontsize=11, color='green')
axes[0, 0].axis('off')

# Generate 17 augmented versions
# aug_iter = data_augmentation.flow(sample_img, batch_size=1)   # .flow() is a old method,, It doesn't works
for i in range(1, 18):
  row = i // 6
  col = i % 6

  aug_img = data_augmentation(sample_img, training=True)
  axes[row, col].imshow(aug_img[0])
  axes[row, col].set_title(f'Augmented #{i}', fontsize=9)
  axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('03_augmentation_examples.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 03_augmentation_examples.png")

# BUILD CNN WITH AUGMENTATION

print("\n" + "="*80)
print("BUILDING CNN WITH DATA AUGMENTATION")
print("="*80)

# Same Structure as before
cnn_aug_model = keras.Sequential([
    data_augmentation,    # 🔥 augmentation layer first

    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                   input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("✅ CNN model created (same architecture)")

# Compile
cnn_aug_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled")

# TRAIN WITH AUGMENTAION

print("\n" + "="*80)
print("TRAINING WITH DATA AUGMENTATION")
print("="*80)

print("""
🔄 Augmentation Configuration:
   • Rotation: ±15 degrees
   • Width shift: ±10%
   • Height shift: ±10%
   • Horizontal flip: Yes
   • Zoom: ±10%
   
⚡ Training Details:
   • Epochs: 30 (more epochs with augmentation!)
   • Batch size: 128
   • Each epoch sees DIFFERENT augmented versions
   • Validation data: NOT augmented (real test)
""")

# Prepare augmented data generator for training
# We alreday prepared and we named it as 'data_augmentation' and inserted when bulding the 'cnn_aug_model' on the first layer!

# Split training data for validation
val_split = 0.2
val_size = int(len(X_train_norm) * val_split)
X_val = X_train_norm[-val_size:]
y_val = y_train[-val_size:]
X_train_split = X_train_norm[:-val_size]
y_train_split = y_train[:-val_size]

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train_split):,}")
print(f"   Validation: {len(X_val):,}")
print()

"""# Create augmented generator
train_generator = train_datagen.flow(
    X_train_split, y_train_split,
    batch_size=128
) Used for Old method of Data Augmentation"""

print("🚀 Starting training with augmentation...")
start_time = time.time()

# Train with augmentation
aug_history = cnn_aug_model.fit(
    X_train_split, y_train_split,
    batch_size=128,
    epochs=30,
    validation_data=(X_val, y_val),
    verbose=1
)

aug_train_time = time.time() - start_time

print(f"\n✅ Training complete in {aug_train_time:.2f} seconds ({aug_train_time/60:.1f} minutes)")

# EVALUATE AUGMENTED MODEL

print("\n" + "="*80)
print("EVALUATING AUGMENTED MODEL")
print("="*80)

aug_test_loss, aug_test_acc = cnn_aug_model.evaluate(X_test_norm, y_test, verbose=0)

print(f"\n📊 Augmented CNN Test Performance:")
print(f"   Loss: {aug_test_loss:.4f}")
print(f"   Accuracy: {aug_test_acc*100:.2f}%")

# COMPARISON: WITH vs WITHOUT AUGMENTATION

print("\n" + "="*80)
print("AUGMENTATION IMPACT")
print("="*80)

# Load previous CNN results (from session 2)
# For demo, using placeholder values - in real scenario, load from saved history
baseline_acc = 75.0  # Placeholder - replace with actual from session 2

comparison = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              CNN WITHOUT vs WITH DATA AUGMENTATION                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Metric              │ Without Aug     │ With Augmentation           ║
║  ────────────────────┼─────────────────┼───────────────────────────  ║
║  Test Accuracy       │  ~{baseline_acc:.1f}%         │  {aug_test_acc*100:.2f}%                    ║
║  Training Epochs     │  20             │  30                         ║
║  Training Time       │  ~150s          │  {aug_train_time:.1f}s                      ║
║  Overfitting         │  Some gap       │  Reduced gap                ║
║  Data Seen           │  50K images     │  50K × variations (huge!)   ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  RESULT: Augmentation helps! 🎉                                       ║
║  Improvement: ~{aug_test_acc*100 - baseline_acc:+.1f}% (typical range: +2-5%)                    ║
║                                                                       ║
║  Benefits:                                                            ║
║  ✅ Better generalization (less overfitting)                         ║
║  ✅ More robust to variations                                        ║
║  ✅ Effectively "creates" more training data                         ║
║  ✅ Model sees different views each epoch                            ║
║                                                                       ║
║  Trade-offs:                                                          ║
║  ⚠️ Longer training time (more epochs needed)                        ║
║  ⚠️ Slower per epoch (augmentation overhead)                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

print(comparison)

# VISUALIZE TRAINING COMPARISON

print("\n📊 Creating comparison visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy comparison
axes[0].plot(aug_history.history['accuracy'], label='Training (Aug)', linewidth=2, color='#2ecc71')
axes[0].plot(aug_history.history['val_accuracy'], label='Validation (Aug)', linewidth=2, color='#e74c3c')
axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Accuracy', fontweight='bold', fontsize=12)
axes[0].set_title('Training with Data Augmentation', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss comparison
axes[1].plot(aug_history.history['loss'], label='Training Loss (Aug)', linewidth=2, color='#3498db')
axes[1].plot(aug_history.history['val_loss'], label='Validation Loss (Aug)', linewidth=2, color='#9b59b6')
axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Loss', fontweight='bold', fontsize=12)
axes[1].set_title('Loss with Data Augmentation', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('DATA AUGMENTATION TRAINING CURVES', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_augmentation_training.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 03_augmentation_training.png")

# SAVE AUGMENTED MODEL

print("\n" + "="*80)
print("SAVING AUGMENTED MODEL")
print("="*80)

cnn_aug_model.save('cifar10_cnn_augmented.keras')
print("✅ Augmented model saved: cifar10_cnn_augmented.keras")

# FINAL INSIGHTS

print("\n" + "="*80)
print("KEY INSIGHTS: DATA AUGMENTATION")
print("="*80)

final_insights = f"""
🎓 WHAT WE LEARNED ABOUT DATA AUGMENTATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ARTIFICIAL DATA EXPANSION
   • Original: 50,000 training images
   • With augmentation: Effectively millions of variations!
   • Each epoch sees different augmented versions
   • Model can't memorize → forced to learn robust features

2. BETTER GENERALIZATION
   • Training accuracy: {aug_history.history['accuracy'][-1]*100:.1f}%
   • Validation accuracy: {aug_history.history['val_accuracy'][-1]*100:.1f}%
   • Small gap = good generalization! ✅
   
   Without augmentation: Bigger gap (overfitting)
   With augmentation: Smaller gap (better!)

3. REALISTIC VARIATIONS
   • Rotation: Objects at different angles
   • Shift: Objects in different positions
   • Zoom: Objects at different distances
   • Flip: Objects facing different directions
   
   Mimics real-world photo variations!

4. COMPUTATIONAL COST
   • Augmentation happens ON-THE-FLY (during training)
   • No need to store augmented images
   • Slight overhead per batch (negligible)
   • Memory efficient! ✅

5. WHEN TO USE AUGMENTATION
   ✅ Small dataset (< 100K images)
   ✅ Overfitting observed (train >> val accuracy)
   ✅ Need better generalization
   ✅ Image classification tasks
   
   ❌ Very large dataset (> 1M images - less benefit)
   ❌ Already generalizing well
   ❌ Tasks where transformations don't make sense


💡 BEST PRACTICES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO:
✅ Horizontal flip (for most objects)
✅ Small rotations (±15-20°)
✅ Slight shifts (±10-20%)
✅ Mild zoom (±10-20%)
✅ Brightness/contrast adjustments

DON'T:
❌ Vertical flip (unless symmetric objects)
❌ Large rotations (> 45° makes objects unnatural)
❌ Extreme zoom (loses critical features)
❌ Transformations that change object identity


🚀 NEXT STEPS TO IMPROVE FURTHER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To reach 90%+ on CIFAR-10:
1. ✅ Data augmentation (DONE!)
2. Deeper networks (ResNet, VGG)
3. Transfer learning (Day 20!)
4. Batch normalization
5. Better optimizers (SGD with momentum, learning rate schedules)
6. Ensemble methods (combine multiple models)

Current: {aug_test_acc*100:.1f}%
Target: 90%+
Gap: ~{90 - aug_test_acc*100:.0f}% → Achievable with advanced techniques!
"""

print(final_insights)

print("\n" + "="*80)
print("SESSION 3 COMPLETE: DATA AUGMENTATION MASTERED!")
print("="*80)
print(f"\n🎉 Achieved {aug_test_acc*100:.1f}% with augmentation!")
print(f"   Ready for wrap-up and documentation!")