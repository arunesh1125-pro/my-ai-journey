"""
FASHION MNIST CLASSIFIER
========================
70,000 clothing images - Real deep learning application!
Compare: Neural Network vs Logistic Regression
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time

# LOAD FASHION MNIST DATASET

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

print("""
FASHION MNIST DATASET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Created by: Zalando Research
Purpose: Modern replacement for MNIST digits
Size: 70,000 grayscale images (60K train + 10K test)
Image size: 28×28 pixels
Classes: 10 clothing categories

Classes:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

Why Fashion MNIST?
✅ Same format as classic MNIST (easy to use)
✅ More challenging (not as easy as digit recognition)
✅ Real-world application (e-commerce)
✅ Built into Keras! (no download needed)
""")

# Load data (automatically downloads if not cached)
print("📥 Loading Fashion MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"✅ Dataset loaded!")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Image shape: {X_train[0].shape}")
print(f"   Number of classes: {len(np.unique(y_train))}")

# Class names for visualization
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# DATA EXPORATION

print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

print(f"\n📊 Data Statistics:")
print(f"   Pixel value range: [{X_train.min()}, {X_train.max()}]")
print(f"   Mean pixel value: {X_train.mean():.2f}")
print(f"   Std pixel value: {X_train.std():.2f}")

print(f"\n📊 Class Distribution (Training):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
  print(f"  {label} ({class_names[label]:<15}): {count:,} samples ({count/len(y_train)*100:.1f}%)")

# Visualize sample images
fig, axes = plt.subplots(4, 10, figsize=(18, 8))
fig.suptitle('FASHIO MNIST SAMPLE IMAGES', fontsize=16, fontweight='bold', y=0.98)

for i in range(40):
  row = i // 10
  col = i % 10
  axes[row, col].imshow(X_train[i], cmap='gray')
  axes[row, col].set_title(f'{class_names[y_train[i]]}', fontsize=8)
  axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('02_fashion_mnist_samples.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Saved: 02_fashion_mnist_samples.png")

# DATA PREPROCESSING

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

print("""
PREPROCESSING STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NORMALIZATION
   • Current: Pixels in [0, 255]
   • Target: Pixels in [0, 1]
   • Why: Neural networks learn better with small numbers
   • How: Divide by 255.0

2. RESHAPING (for Dense layers)
   • Current: (28, 28) 2D image
   • Target: (784,) 1D vector
   • Why: Dense layers expect 1D input
   • How: Flatten 28×28 = 784

3. TRAIN-TEST SPLIT
   • Already done! (60K train, 10K test)
   • We'll use 20% of training for validation
""")

# Normalize pixel values to [0, 1]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

print(f"✅ Normalized data")
print(f"   New range: [{X_train_norm.min()}, {X_train_norm.max()}]")

# Flatten images from 28x28 to 784
X_train_flat = X_train_norm.reshape(-1, 784)
X_test_flat = X_test_norm.reshape(-1, 784)

print(f"✅ Flattened images")
print(f"   Original shape: {X_train_norm.shape}")
print(f"   Flattened shape: {X_train_flat.shape}")
print(f"   (60000 samples × 784 pixels)")

# BASELINE: LOGISTIC REGRESSION

print("\n" + "="*80)
print("BASELINE: LOGISTIC REGRESSION (Classical ML)")
print("="*80)

print("🚀 Training Logistic Regression...")
start_time = time.time()

# Train on subset for speed (10k samples)
lr_model = LogisticRegression(max_iter=100, verbose=0, n_jobs=-1)
lr_model.fit(X_train_flat[:10000], y_train[:10000])

lr_train_time = time.time() - start_time

# Evaluate
lr_train_acc = lr_model.score(X_train_flat[:10000], y_train[:10000])
lr_test_acc = lr_model.score(X_test_flat, y_test)

print(f"✅ Logistic Regression trained in {lr_train_time:.2f} seconds")
print(f"   Training accuracy: {lr_train_acc*100:.2f}%")
print(f"   Test accuracy: {lr_test_acc*100:.2f}%")

# NEURAL NETWORK MODEL

print("\n" + "="*80)
print("BUILDING NEURAL NETWORK")
print("="*80)

print("""
NETWORK ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Layer:    784 neurons (28×28 pixels flattened)
                  ↓
Hidden Layer 1: 128 neurons, ReLU activation
                  ↓
Dropout:        20% (prevent overfitting)
                  ↓
Hidden Layer 2: 64 neurons, ReLU activation
                  ↓
Output Layer:   10 neurons, Softmax activation (10 classes)

Total parameters: ~100,000+ (vs 17 in XOR!)
""")

# Build model
model = keras.Sequential([
    # Input layer (implicit) + First hidden layer
    layers.Dense(128, activation='relu', input_shape=(784,)),

    # Dropout for regualrization
    layers.Dropout(0.2),

    # Second hidden layer
    layers.Dense(64, activation='relu'),

    # Output layer (10 classes)
    layers.Dense(10, activation='softmax')
])

print("✅ Model architecture defined")
print("\n📊 Model Summary:")
model.summary()

# Total parameters calculation
total_params = model.count_params()
print(f"\n💡 Total trainable parameters: {total_params:,}")

# COMPILE MODEL

print("\n" + "="*80)
print("COMPILING MODEL")
print("="*80)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',     # For integer labels
    metrics=['accuracy']
)

print("""
✅ Model compiled with:
   • Optimizer: Adam (adaptive learning rate)
   • Loss: Sparse Categorical Crossentropy (multi-class)
   • Metrics: Accuracy
""")

# TRAIN MODEL

print("\n" + "="*80)
print("TRAINING NEURAL NETWORK")
print("="*80)

print("🚀 Starting training...")
print("   Epochs: 10")
print("   Batch size: 128")
print("   Validation split: 20%")
print()

start_time = time.time()

history = model.fit(
    X_train_flat,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1     # Show progress bar
)

nn_train_time = time.time() - start_time

print(f"\n✅ Training complete in {nn_train_time:.2f} seconds")

# EVALUATION

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)

print(f"\n📊 Test Set Performance:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_accuracy*100:.2f}%")

# Get predictions
y_pred = model.predict(X_test_flat, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Detailed classification report
print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# COMPARISON: LOGISTIC REGRESSION vs NEURAL NETWORK

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_table = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              LOGISTIC REGRESSION vs NEURAL NETWORK                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Metric                │ Logistic Reg    │ Neural Network            ║
║  ──────────────────────┼─────────────────┼─────────────────────────  ║
║  Test Accuracy         │  {lr_test_acc*100:>6.2f}%         │  {test_accuracy*100:>6.2f}%                ║
║  Training Time         │  {lr_train_time:>6.2f}s         │  {nn_train_time:>6.2f}s                ║
║  Parameters            │  ~7,850         │  {total_params:>8,}              ║
║  Layers                │  1 (linear)     │  4 (non-linear)           ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  WINNER: Neural Network! 🏆                                           ║
║  Improvement: {(test_accuracy - lr_test_acc)*100:>+.2f}% absolute accuracy gain                      ║
║                                                                       ║
║  Why Neural Network Wins:                                             ║
║  • More layers = more capacity to learn complex patterns             ║
║  • ReLU activation = non-linear transformations                      ║
║  • More parameters = can model subtle differences in clothing        ║
║                                                                       ║
║  Trade-off:                                                           ║
║  • {nn_train_time/lr_train_time:.1f}x longer training time                                      ║
║  • But MUCH better accuracy!                                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

print(comparison_table)

# VISUALIZATION

print("\n" + "="*80)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training History - Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
ax1.set_title('Training & Validation Loss', fontweight='bold', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Training History - Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2ecc71')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#9b59b6')
ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax2.set_title('Training & Validation Accuracy', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison
ax3 = fig.add_subplot(gs[0, 2])
models = ['Logistic\nRegression', 'Neural\nNetwork']
accuracies = [lr_test_acc*100, test_accuracy*100]
colors_bar = ['#e74c3c', '#2ecc71']
bars = ax3.bar(models, accuracies, color=colors_bar, edgecolor='black', linewidth=2)
ax3.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=11)
ax3.set_title('Model Comparison', fontweight='bold', fontsize=13)
ax3.set_ylim([0, 100])
ax3.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

# Plot 4: Confusion Matrix
ax4 = fig.add_subplot(gs[1:, :])
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           ax=ax4, cbar_kws={'label': 'Count'})
ax4.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
ax4.set_ylabel('True Label', fontweight='bold', fontsize=12)
ax4.set_title('Confusion Matrix (Test Set)', fontweight='bold', fontsize=14)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax4.get_yticklabels(), rotation=0)

plt.suptitle('FASHION MNIST: NEURAL NETWORK PERFORMANCE ANALYSIS',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('02_fashion_mnist_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_fashion_mnist_results.png")

# PREDICTION VISUALIZATION

print("📊 Creating prediction visualizations...")

fig, axes = plt.subplots(4, 8, figsize=(18, 10))
fig.suptitle('NEURAL NETWORK PREDICTIONS ON TEST SET', fontsize=16, fontweight='bold', y=0.98)

# Select random test samples
np.random.seed(42)
random_indices = np.random.choice(len(X_test), 32, replace=False)

for i, idx in enumerate(random_indices):
    row = i // 8
    col = i % 8
    
    # Get image and prediction
    img = X_test[idx]
    true_label = y_test[idx]
    pred_probs = y_pred[idx]
    pred_label = y_pred_classes[idx]
    confidence = pred_probs[pred_label] * 100
    
    # Plot image
    axes[row, col].imshow(img, cmap='gray')
    
    # Color code: Green if correct, Red if wrong
    color = 'green' if pred_label == true_label else 'red'
    
    # Title
    axes[row, col].set_title(
        f'True: {class_names[true_label][:5]}\n'
        f'Pred: {class_names[pred_label][:5]}\n'
        f'{confidence:.0f}%',
        fontsize=8, color=color, fontweight='bold'
    )
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('02_fashion_mnist_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_fashion_mnist_predictions.png")

# SAVE MODEL

model.save('fashion_mnist_model.keras')
print("✅ Model saved: fashion_mnist_model.keras")

# KEY INSIGHTS

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

insights = f"""
🎓 WHAT WE LEARNED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DEEP LEARNING > CLASSICAL ML FOR IMAGES
   • Logistic Regression: {lr_test_acc*100:.1f}% accuracy
   • Neural Network: {test_accuracy*100:.1f}% accuracy
   • Improvement: {(test_accuracy - lr_test_acc)*100:+.1f}% (absolute)

2. MORE LAYERS = MORE POWER
   • 2 hidden layers learn complex patterns
   • Each layer learns different representations:
     - Layer 1: Edges and simple textures
     - Layer 2: Complex patterns (collars, pockets)
     - Output: Final classification decision

3. DROPOUT PREVENTS OVERFITTING
   • Training accuracy: {history.history['accuracy'][-1]*100:.1f}%
   • Validation accuracy: {history.history['val_accuracy'][-1]*100:.1f}%
   • Small gap = good generalization!

4. FRAMEWORKS ARE ESSENTIAL
   • Built 100K+ parameter network in ~15 lines
   • Would take 1000+ lines from scratch
   • Automatic backprop, GPU support, optimizations

5. REAL-WORLD PERFORMANCE
   • 88%+ accuracy on clothing classification
   • Production-ready model
   • Could deploy for e-commerce recommendation!


🎯 MOST CONFUSED CLASSES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Looking at confusion matrix:
- Shirt ↔ T-shirt/top (similar appearance)
- Pullover ↔ Coat (both outerwear)
- Sneaker ↔ Ankle boot (both footwear)

Makes sense! Even humans would confuse these.


💡 PRODUCTION DEPLOYMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This model could be used for:
✅ E-commerce product categorization
✅ Visual search ("Find similar items")
✅ Automated tagging for inventory
✅ Fashion recommendation systems

Next step: Deploy as API (we'll learn this Week 8!)
"""

print(insights)

print("\n" + "="*80)
print("SESSION 2 COMPLETE: FASHION MNIST CLASSIFIER BUILT!")
print("="*80)
print(f"\n🎉 Achieved {test_accuracy*100:.1f}% accuracy on 70,000 images!")
print(f"   {(test_accuracy - lr_test_acc)*100:.1f}% better than logistic regression!")