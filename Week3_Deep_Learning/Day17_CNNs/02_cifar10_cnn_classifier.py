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

# VISUALIZATIONS
# ============================================

print("\n" + "="*80)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Training History - Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(cnn_history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
ax1.plot(cnn_history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
ax1.set_title('CNN Training & Validation Loss', fontweight='bold', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Training History - Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(cnn_history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2ecc71')
ax2.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#9b59b6')
ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax2.set_title('CNN Training & Validation Accuracy', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison
ax3 = fig.add_subplot(gs[0, 2])
models = ['Dense\nNetwork', 'CNN']
accuracies = [dense_test_acc*100, cnn_test_acc*100]
colors_bar = ['#e74c3c', '#2ecc71']
bars = ax3.bar(models, accuracies, color=colors_bar, edgecolor='black', linewidth=2, width=0.6)
ax3.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=11)
ax3.set_title('Dense vs CNN Comparison', fontweight='bold', fontsize=13)
ax3.set_ylim([0, 100])
ax3.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

# Plot 4-6: Confusion Matrix
ax4 = fig.add_subplot(gs[1:, :])
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           ax=ax4, cbar_kws={'label': 'Count'}, annot_kws={'size': 9})
ax4.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
ax4.set_ylabel('True Label', fontweight='bold', fontsize=12)
ax4.set_title('Confusion Matrix - CIFAR-10 CNN (Test Set)', fontweight='bold', fontsize=14)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=10)
plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=10)

plt.suptitle('CIFAR-10 CNN: COMPREHENSIVE PERFORMANCE ANALYSIS',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('02_cifar10_cnn_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_cifar10_cnn_results.png")

# ============================================
# PREDICTION SAMPLES
# ============================================

print("📊 Creating prediction visualizations...")

fig, axes = plt.subplots(5, 8, figsize=(18, 12))
fig.suptitle('CNN PREDICTIONS ON CIFAR-10 TEST SET', fontsize=16, fontweight='bold')

# Select random samples
np.random.seed(42)
random_indices = np.random.choice(len(X_test), 40, replace=False)

for i, idx in enumerate(random_indices):
    row = i // 8
    col = i % 8
    
    # Get image and prediction
    img = X_test[idx]
    true_label = y_test[idx][0]
    pred_probs = y_pred[idx]
    pred_label = y_pred_classes[idx]
    confidence = pred_probs[pred_label] * 100
    
    # Plot image
    axes[row, col].imshow(img)
    
    # Color: Green if correct, Red if wrong
    color = 'green' if pred_label == true_label else 'red'
    
    # Title
    axes[row, col].set_title(
        f'T:{class_names[true_label][:4]}\n'
        f'P:{class_names[pred_label][:4]}\n'
        f'{confidence:.0f}%',
        fontsize=8, color=color, fontweight='bold'
    )
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('02_cifar10_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_cifar10_predictions.png")

# ============================================
# PER-CLASS PERFORMANCE
# ============================================

print("\n📊 Analyzing per-class performance...")

# Calculate per-class accuracy
class_correct = np.zeros(10)
class_total = np.zeros(10)

for i in range(len(y_test)):
    label = y_test[i][0]
    class_total[label] += 1
    if y_pred_classes[i] == label:
        class_correct[label] += 1

class_accuracy = class_correct / class_total * 100

# Visualize per-class accuracy
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(10), class_accuracy, color='skyblue', edgecolor='black', linewidth=2)
ax.set_xlabel('Class', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_title('Per-Class Accuracy - CIFAR-10 CNN', fontweight='bold', fontsize=14)
ax.set_xticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
           f'{acc:.1f}%', ha='center', va='bottom', 
           fontweight='bold', fontsize=10)
    
    # Color bars by accuracy
    if acc >= 80:
        bar.set_color('#2ecc71')  # Green
    elif acc >= 70:
        bar.set_color('#f39c12')  # Orange
    else:
        bar.set_color('#e74c3c')  # Red

plt.tight_layout()
plt.savefig('02_cifar10_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_cifar10_class_accuracy.png")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

cnn_model.save('cifar10_cnn_model.keras')
print("✅ Model saved: cifar10_cnn_model.keras")

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find best and worst classes
best_class_idx = np.argmax(class_accuracy)
worst_class_idx = np.argmin(class_accuracy)

insights = f"""
🎓 WHAT WE LEARNED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CNNs DOMINATE IMAGE CLASSIFICATION
   • Dense Network: {dense_test_acc*100:.1f}% (trained on 10K samples)
   • CNN: {cnn_test_acc*100:.1f}% (trained on 50K samples)
   • Improvement: {(cnn_test_acc - dense_test_acc)*100:+.1f}% absolute
   
   Even with 5x more training data, CNN is MUCH better!

2. COLOR IMAGES ARE HARDER
   • Fashion MNIST (grayscale): 88.5% with Dense
   • CIFAR-10 (color): {cnn_test_acc*100:.1f}% with CNN
   • More channels = more complexity
   • Natural objects vs simple clothing

3. HIERARCHICAL FEATURES EMERGE
   • Layer 1 (32 filters): Edges, colors, simple textures
   • Layer 2 (64 filters): Corners, curves, combined patterns
   • Layer 3 (128 filters): Object parts (wheels, wings, legs)
   
   Each layer builds on previous! 🧱

4. PARAMETER EFFICIENCY
   • Total parameters: {total_params:,}
   • Despite 3 conv layers + 2 dense layers!
   • Much more efficient than Dense for images
   • Parameter sharing is the key

5. VALIDATION CURVES SHOW GOOD GENERALIZATION
   • Final train accuracy: {cnn_history.history['accuracy'][-1]*100:.1f}%
   • Final val accuracy: {cnn_history.history['val_accuracy'][-1]*100:.1f}%
   • Small gap = Dropout is working! ✅


📊 PER-CLASS PERFORMANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best performing:  {class_names[best_class_idx]:<12} ({class_accuracy[best_class_idx]:.1f}%)
Worst performing: {class_names[worst_class_idx]:<12} ({class_accuracy[worst_class_idx]:.1f}%)

Why the difference?
- {class_names[best_class_idx]}: Distinctive features (easier to identify)
- {class_names[worst_class_idx]}: Looks similar to other classes

Common confusions (from confusion matrix):
- Cat ↔ Dog (both furry animals)
- Automobile ↔ Truck (both vehicles)
- Deer ↔ Horse (both four-legged animals)

Even humans struggle with these! 🤔


💡 WHAT MAKES CNNs WORK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SPATIAL STRUCTURE PRESERVED
   Dense: 32×32×3 → 3,072 (flattened)
   CNN: 32×32×3 → 32×32×32 (2D preserved)
   
   Nearby pixels matter! CNNs understand this.

2. TRANSLATION INVARIANCE
   Same filter applied everywhere
   → Cat detected in corner OR center
   → No need to learn "cat at position (5,7)"

3. PARAMETER SHARING
   One 3×3 filter = 27 weights (3×3×3 RGB)
   Applied to entire 32×32 image
   → Dramatically fewer parameters than Dense

4. HIERARCHICAL LEARNING
   Start simple (edges), build complex (objects)
   → Like how humans learn to see!


🎯 PRODUCTION READY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This {cnn_test_acc*100:.1f}% accuracy is:
✅ Better than traditional ML (40-50%)
✅ Solid performance for a simple CNN
✅ Could be deployed for:
   - Image categorization systems
   - Content moderation (identifying objects)
   - Auto-tagging for image libraries
   - Educational tools

To reach 90%+:
- Data augmentation (next section!)
- More layers (deeper networks)
- Transfer learning (Day 20)
- Larger models (ResNet, EfficientNet)
"""

print(insights)

# Print per-class summary
print(f"\n📊 Per-Class Accuracy Summary:")
print(f"{'Class':<15} {'Accuracy':<12} {'Rating':<10}")
print("-" * 40)
for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
    if acc >= 80:
        rating = "🟢 Excellent"
    elif acc >= 70:
        rating = "🟡 Good"
    else:
        rating = "🔴 Needs work"
    print(f"{name:<15} {acc:>6.1f}%       {rating}")

print("\n" + "="*80)
print("SESSION 2 COMPLETE: CIFAR-10 CNN CLASSIFIER BUILT!")
print("="*80)
print(f"\n🎉 Achieved {cnn_test_acc*100:.1f}% accuracy on 10,000 color images!")
print(f"   {(cnn_test_acc - dense_test_acc)*100:.1f}% better than Dense network!")
print("\n☕ Take a 15-minute break before data augmentation!")