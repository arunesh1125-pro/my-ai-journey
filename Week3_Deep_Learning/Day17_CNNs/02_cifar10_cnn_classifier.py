"""
CIFAR-10 CNN CLASSIFIER
=======================
60,000 color images across 10 classes
Proving CNNs are superior for image recognition
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import time

print("="*80)
print("CIFAR-10: CNN FOR COLOR IMAGES")
print("="*80)

# LOAD CIFAR-10 DATASET

print("\n" + "="*80)
print("LOADING CIFAR-10 DATASET")
print("="*80)

print("""
CIFAR-10 DATASET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Created by: University of Toronto
Purpose: Real-world object recognition benchmark
Size: 60,000 color images (50K train + 10K test)
Image size: 32×32 pixels, 3 channels (RGB)
Classes: 10 categories

Classes:
0: Airplane ✈️
1: Automobile 🚗
2: Bird 🐦
3: Cat 🐱
4: Deer 🦌
5: Dog 🐕
6: Frog 🐸
7: Horse 🐴
8: Ship 🚢
9: Truck 🚚

Difficulty: MUCH harder than Fashion MNIST!
- Color images (3 channels vs 1)
- Natural objects (vs simple clothing)
- More variation within classes
- Background clutter

Benchmark Accuracies:
- Random guessing: 10%
- Traditional ML: 40-50%
- Simple CNN: 70-75%
- Advanced CNN: 90%+
- Human performance: ~94%

Our goal today: 75%+ with simple CNN! 🎯
""")

# Load data
print("📥 Loading CIFAR-10...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print(f"✅ Dataset loaded!")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Image shape: {X_train[0].shape}")
print(f"   Number of classes: {len(np.unique(y_train))}")

# Class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# DATA EXPLORATION

print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

print(f"\n📊 Data Statistics:")
print(f"   Pixel value range: [{X_train.min()}, {X_train.max()}]")
print(f"   Mean pixel value: {X_train.mean():.2f}")
print(f"   Image dimensions: {X_train.shape[1]}×{X_train.shape[2]}")
print(f"   Color channels: {X_train.shape[3]} (RGB)")

print(f"\n📊 Class Distribution (Training):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
  print(f"   {label} ({class_names[label]:<12}): {count:,} samples ({count/len(y_train)*100:.1f}%)")

# Visualize samples
print("\n📊 Creating sample visualization...")

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
fig.suptitle('CIFAR-10 SAMPLE IMAGES (10 per class)', fontsize=16, fontweight='bold')

for class_idx in range(10):
    # Get 10 samples from this class
    class_samples = X_train[y_train.flatten() == class_idx][:10]
    
    for sample_idx in range(10):
        ax = axes[class_idx, sample_idx]
        ax.imshow(class_samples[sample_idx])
        
        if sample_idx == 0:
            ax.set_ylabel(class_names[class_idx], fontweight='bold', fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('02_cifar10_samples.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_cifar10_samples.png")

# ============================================
# DATA PREPROCESSING
# ============================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

print("""
PREPROCESSING FOR CNNs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NORMALIZATION
   • Current: Pixels in [0, 255]
   • Target: Pixels in [0, 1]
   • Why: Neural networks train better with small numbers
   • How: Divide by 255.0

2. SHAPE (for CNNs)
   • Keep 2D structure! (32, 32, 3)
   • No flattening needed for Conv layers
   • Channels last: (height, width, channels)

3. LABELS
   • Already integers [0-9]
   • Perfect for sparse_categorical_crossentropy
   • No need to flatten (Keras handles it)
""")

# Normalize to [0, 1]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm  = X_test.astype('float32') / 255.0

print(f"✅ Normalized data")
print(f"   New range: [{X_train_norm.min()}, {X_train_norm.max()}]")
print(f"   Shape preserved: {X_train_norm.shape}")

# BASELINE: DENSE NETWORK (Will Fail!)

print("\n" + "="*80)
print("BASELINE: DENSE NETWORK (For Comparison)")
print("="*80)

print("🔨 Building Dense network (like Day 16)...")

# Build Dense Network
dense_model = keras.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),    # 32x32x3 = 3,072 inputs
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("\n📊 Dense Model Summary:")
dense_model.summary()

# compile
dense_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train (small subset for speed)
print("\n🚀 Training Dense network (on 10K samples for speed)...")
start_time = time.time()

dense_history = dense_model.fit(
    X_train_norm[:10000], y_train[:10000],
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

dense_train_time = time.time() - start_time

# Evaluate
dense_test_loss, dense_test_acc = dense_model.evaluate(X_test_norm, y_test, verbose=0)

print(f"\n✅ Dense Network Results:")
print(f"   Training time: {dense_train_time:.2f} seconds")
print(f"   Test accuracy: {dense_test_acc*100:.2f}%")
print(f"   (Trained on only 10K samples)")

# CNN MODEL

print("\n" + "="*80)
print("BUILDING CNN MODEL")
print("="*80)

print("""
CNN ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:           32×32×3     (RGB color image)
                    ↓
Conv2D(32):      32×32×32    (32 filters, 3×3, padding='same')
ReLU:            32×32×32
MaxPool2D:       16×16×32    (2×2 pool → halves dimensions)
                    ↓
Conv2D(64):      16×16×64    (64 filters, 3×3, padding='same')
ReLU:            16×16×64
MaxPool2D:       8×8×64      (2×2 pool → halves dimensions)
                    ↓
Conv2D(128):     8×8×128     (128 filters, 3×3, padding='same')
ReLU:            8×8×128
MaxPool2D:       4×4×128     (2×2 pool → halves dimensions)
                    ↓
Flatten:         2,048       (4×4×128 = 2,048)
                    ↓
Dense(128):      128         (fully connected)
ReLU:            128
Dropout(0.5):    128         (prevent overfitting)
                    ↓
Dense(10):       10          (output layer)
Softmax:         10          (probabilities)

Key Design Choices:
✅ 3 Conv blocks (more depth = better features)
✅ Filters double after each pool (32→64→128)
✅ 3×3 filters (standard, works well)
✅ 'same' padding (preserve dimensions)
✅ MaxPooling after each conv block
✅ Dropout 0.5 (strong regularization for small dataset)
""")

# Build CNN
cnn_model = keras.Sequential([
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

    # Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("✅ CNN model created!\n")
print("📊 CNN Model Summary:")
cnn_model.summary()

# Count parameters
total_params = cnn_model.count_params()
print(f"\n💡 Total trainable parameters: {total_params:,}")

# COMPILE CNN

print("\n" + "="*80)
print("COMPILING CNN")
print("="*80)

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("""
✅ Model compiled with:
   • Optimizer: Adam
   • Loss: Sparse Categorical Crossentropy
   • Metrics: Accuracy
""")

# TRAIN CNN

print("\n" + "="*80)
print("TRAINING CNN")
print("="*80)

print("🚀 Starting training...")
print("   Epochs: 20")
print("   Batch size: 128")
print("   Validation split: 20%")
print("   Training samples: 50,000")
print()

start_time = time.time()

cnn_history = cnn_model.fit(
    X_train_norm, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

cnn_train_time = time.time() - start_time

print(f"\n✅ Training complete in {cnn_train_time:.2f} seconds ({cnn_train_time/60:.1f} minutes)")

# EVALUATION
# ============================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Evaluate on test set
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_norm, y_test, verbose=0)

print(f"\n📊 CNN Test Set Performance:")
print(f"   Loss: {cnn_test_loss:.4f}")
print(f"   Accuracy: {cnn_test_acc*100:.2f}%")

# Get predictions
y_pred = cnn_model.predict(X_test_norm, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Detailed report
print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# ============================================
# COMPARISON
# ============================================

print("\n" + "="*80)
print("MODEL COMPARISON: DENSE vs CNN")
print("="*80)

comparison = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    DENSE NETWORK vs CNN                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Metric              │ Dense Network   │ CNN                         ║
║  ────────────────────┼─────────────────┼───────────────────────────  ║
║  Test Accuracy       │  {dense_test_acc*100:>6.2f}%         │  {cnn_test_acc*100:>6.2f}%                   ║
║  Training Samples    │  10,000         │  50,000                     ║
║  Training Time       │  {dense_train_time:>6.1f}s         │  {cnn_train_time:>6.1f}s ({cnn_train_time/60:.1f}min)          ║
║  Parameters          │  ~397K          │  {total_params:>6,}                   ║
║  Architecture        │  Flatten+Dense  │  Conv+Pool+Dense            ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  WINNER: CNN! 🏆                                                      ║
║  Improvement: {(cnn_test_acc - dense_test_acc)*100:>+.1f}% absolute accuracy gain                        ║
║                                                                       ║
║  Why CNN Wins on Images:                                              ║
║  ✅ Preserves spatial structure (2D → 2D)                            ║
║  ✅ Learns hierarchical features (edges → objects)                   ║
║  ✅ Translation invariance (object anywhere in image)                ║
║  ✅ Fewer connections but smarter (parameter sharing)                ║
║                                                                       ║
║  Dense network treats pixels independently → loses spatial info      ║
║  CNN understands pixels are neighbors → captures patterns!           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

print(comparison)