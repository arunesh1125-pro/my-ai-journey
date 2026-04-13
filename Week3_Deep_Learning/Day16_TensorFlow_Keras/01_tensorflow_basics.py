"""
TENSORFLOW/KERAS FUNDAMENTALS
==============================
From manual implementation to framework magic!
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

print("="*80)
print("TENSORFLOW/KERAS INTRODUCTION")
print("="*80)

print(f"\n✅ TensorFlow version: {tf.__version__}")
print(f"✅ Keras version: {keras.__version__}")
print(f"✅ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# WHY TENSORFLOW?

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    WHY USE TENSORFLOW/KERAS?                           ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  YESTERDAY (From Scratch):                                             ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │ • Implemented forward propagation manually (~20 lines)           │ ║
║  │ • Implemented backpropagation manually (~30 lines)               │ ║
║  │ • Implemented gradient descent manually (~10 lines)              │ ║
║  │ • Calculated gradients by hand (complex!)                        │ ║
║  │ • Limited to small networks (XOR: 2→4→1)                         │ ║
║  │ • No GPU support                                                 │ ║
║  │                                                                  │ ║
║  │ Total: ~100 lines for simple network                             │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  TODAY (TensorFlow/Keras):                                             ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │ • Define layers: 1 line each                                     │ ║
║  │ • Compile model: 1 line                                          │ ║
║  │ • Train model: 1 line                                            │ ║
║  │ • Backprop: AUTOMATIC! ✨                                        │ ║
║  │ • Gradients: AUTOMATIC! ✨                                       │ ║
║  │ • GPU support: AUTOMATIC! ✨                                     │ ║
║  │                                                                  │ ║
║  │ Total: ~10 lines for complex network                             │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  BENEFITS:                                                             ║
║  ✅ 10x less code                                                      ║
║  ✅ No manual gradient calculations                                   ║
║  ✅ GPU acceleration (100x faster!)                                   ║
║  ✅ Built-in optimizers (Adam, SGD, RMSprop)                          ║
║  ✅ Pre-built layers (Dense, Conv2D, LSTM)                            ║
║  ✅ Easy model saving/loading                                         ║
║  ✅ Production-ready                                                  ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# TENSORSLOW vs PYTORCH vs KERAS

print("\n" + "="*80)
print("FRAMEWORK LANDSCAPE")
print("="*80)

print("""
POPULAR DEEP LEARNING FRAMEWORKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TENSORFLOW + KERAS (What we're using!) ⭐
   • Developed by: Google
   • Best for: Production deployment, industry use
   • Pros: 
     - Keras API is beginner-friendly
     - TensorFlow Lite (mobile deployment)
     - TensorFlow.js (web deployment)
     - Excellent documentation
   • Cons: 
     - Historically more complex (TF 1.x)
     - Debugging can be harder
   • Used by: Google, Airbnb, Coca-Cola, Twitter

2. PYTORCH
   • Developed by: Meta/Facebook
   • Best for: Research, experimentation
   • Pros:
     - Very pythonic, intuitive
     - Excellent for debugging (eager execution)
     - Popular in academia
   • Cons:
     - Historically less production-ready
     - Smaller deployment ecosystem
   • Used by: Meta, Tesla, Microsoft

3. JAX
   • Developed by: Google Research
   • Best for: High-performance computing, research
   • Pros:
     - Fastest training (XLA compilation)
     - Functional programming style
   • Cons:
     - Steeper learning curve
     - Smaller community

WHAT IS KERAS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keras is a HIGH-LEVEL API for building neural networks.

Think of it like this:
- TensorFlow = Low-level programming (C++)
- Keras = High-level programming (Python)

Keras runs ON TOP of TensorFlow (and previously Theano, CNTK).

Since TensorFlow 2.0 (2019):
→ Keras is BUILT INTO TensorFlow! ✅
→ tensorflow.keras is the recommended way to use TensorFlow

We'll use: tensorflow as tf
keras (best of both worlds!)
""")

# KERAS SEQUENTIAL API

print("\n" + "="*80)
print("KERAS SEQUENTIAL API: BUILDING BLOCKS")
print("="*80)

print("""
TWO WAYS TO BUILD MODELS IN KERAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SEQUENTIAL API (Simple, Linear Models) ← WE START HERE
   • Stack layers one after another
   • Easy to understand
   • 90% of use cases

2. FUNCTIONAL API (Complex, Non-Linear Models)
   • Multiple inputs/outputs
   • Skip connections
   • Advanced architectures


SEQUENTIAL API EXAMPLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Yesterday's XOR network from scratch: ~100 lines

Today with Keras: 10 lines! ⬇️
""")

# Build the SAME XOR network from yesterday, but with Keras!
print("\n🔨 Building XOR Network with Keras:")

model = keras.Sequential([
    layers.Dense(4, activation="sigmoid", input_shape=(2,)),  # Hidden layer
    layers.Dense(1, activation='sigmoid')                      # Output layer
])

print("✅ Model created in 3 lines!")

# Print model architecture
print("\n📊 Model Architecture:")
model.summary()

# UNDERSTANDING LAYERS

print("\n" + "="*80)
print("KERAS LAYERS EXPLAINED")
print("="*80)

print("""
DENSE LAYER (Fully Connected):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layers.Dense(units, activation, input_shape)

Parameters:
- units: Number of neurons in this layer
- activation: Activation function ('relu', 'sigmoid', 'tanh', 'softmax')
- input_shape: Shape of input data (only for FIRST layer)

Example:
layers.Dense(4, activation='relu', input_shape=(2,))

Means:
- 4 neurons in this layer
- ReLU activation
- Expects input with 2 features


COMMON ACTIVATION FUNCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Activation          Use Case                    Output Range
────────────────────────────────────────────────────────────────────────
'relu'              Hidden layers (MOST COMMON) [0, ∞)
'sigmoid'           Binary classification        [0, 1]
'tanh'              Hidden layers (alternative)  [-1, 1]
'softmax'           Multi-class classification   [0, 1] (sum=1)
'linear'/None       Regression                   (-∞, ∞)


INPUT SHAPE EXPLAINED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

input_shape = (features,)  ← Note the comma!

Examples:
- XOR: input_shape=(2,)           → 2 features (A, B)
- MNIST: input_shape=(784,)       → 784 pixels (28×28 flattened)
- Iris: input_shape=(4,)          → 4 features (sepal/petal dimensions)
- Images: input_shape=(28, 28, 1) → 28×28 grayscale image

Only needed for FIRST layer (Keras infers the rest!)
""")

# MODEL COMPILATION

print("\n" + "="*80)
print("MODEL COMPILATION: CHOOSING THE LEARNING STRATEGY")
print("="*80)

print("""
model.compile(optimizer, loss, metrics)

THREE KEY COMPONENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. OPTIMIZER: How to update weights
   • 'adam'      ⭐ MOST POPULAR (adaptive learning rate)
   • 'sgd'          Stochastic Gradient Descent (classic)
   • 'rmsprop'      Good for RNNs
   • 'adagrad'      Good for sparse data

2. LOSS FUNCTION: What to minimize
   • 'binary_crossentropy'      → Binary classification (0/1)
   • 'categorical_crossentropy' → Multi-class (one-hot encoded)
   • 'sparse_categorical_crossentropy' → Multi-class (integer labels)
   • 'mse' (mean_squared_error) → Regression

3. METRICS: What to track (for humans, not training)
   • ['accuracy']               → Classification accuracy
   • ['mae']                    → Mean Absolute Error (regression)
   • ['precision', 'recall']    → Advanced metrics


EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Binary Classification (e.g., spam detection):
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

Multi-class Classification (e.g., digit recognition):
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

Regression (e.g., house prices):
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
""")

# Compile our XOR model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Model compiled!")

# TRAIN THE MODEL

print("\n" + "="*80)
print("TRAINING: THE MAGIC HAPPENS HERE")
print("="*80)

print("""
model.fit(X, y, epochs, batch_size, validation_split, verbose)

PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- X: Training data (features)
- y: Training labels (targets)
- epochs: How many times to see entire dataset
- batch_size: How many samples before updating weights
- validation_split: Fraction of data for validation (e.g., 0.2 = 20%)
- verbose: 
    0 = silent
    1 = progress bar
    2 = one line per epoch


UNDERSTANDING BATCH SIZE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: 1000 samples
Batch size: 32

Process:
1. Take first 32 samples
2. Forward pass (make predictions)
3. Calculate loss
4. Backprop & update weights
5. Take next 32 samples
6. Repeat...

One epoch = 1000/32 = 31.25 ≈ 32 batches

Batch size trade-offs:
- Small (8-32): Noisy gradients, better generalization, slower
- Large (128-256): Smooth gradients, faster, may overfit
- Common: 32 or 64


VALIDATION SPLIT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

validation_split=0.2 means:
- 80% data for training
- 20% data for validation (checking overfitting)

Keras automatically splits for you!
""")

# Train on XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

print("\n🚀 Training XOR model...")

history = model.fit(
    X_xor,
    y_xor,
    epochs=1000,
    batch_size=4,
    verbose=0  # Silent training
)

print(f"✅ Training complete!")
print(f"   Final accuracy: {history.history['accuracy'][-1]*100:.1f}%")
print(f"   Final loss: {history.history['loss'][-1]:.6f}")

# Make predictions
predictions = model.predict(X_xor, verbose=0)

print("\n📊 XOR Predictions:")
print(f"{'Input A':<10} {'Input B':<10} {'True':<10} {'Predicted':<15} {'Rounded':<10}")
print("-" * 55)
for i in range(len(X_xor)):
  pred_val = predictions[i][0]
  rounded = round(pred_val)
  print(f"{int(X_xor[i,0]):<10} {int(X_xor[i,1]):<10} {int(y_xor[i,0]):<10} {pred_val:<15.6f} {rounded:<10}")

# COMPARISON: FROM SCRATCH vs KERAS

print("\n" + "="*80)
print("CODE COMPARISON: YESTERDAY vs TODAY")
print("="*80)

comparison = """
╔════════════════════════════════════════════════════════════════════════╗
║                    YESTERDAY vs TODAY                                  ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  YESTERDAY (NumPy from scratch):                                       ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │ # Initialize weights                                             │ ║
║  │ w1 = np.random.randn(2, 4) * 0.5                                 │ ║
║  │ b1 = np.zeros((1, 4))                                            │ ║
║  │ w2 = np.random.randn(4, 1) * 0.5                                 │ ║
║  │ b2 = np.zeros((1, 1))                                            │ ║
║  │                                                                  │ ║
║  │ # Training loop                                                  │ ║
║  │ for epoch in range(10000):                                       │ ║
║  │     # Forward pass                                               │ ║
║  │     hidden_z = X @ w1 + b1                                       │ ║
║  │     hidden_a = sigmoid(hidden_z)                                 │ ║
║  │     output_z = hidden_a @ w2 + b2                                │ ║
║  │     output = sigmoid(output_z)                                   │ ║
║  │                                                                  │ ║
║  │     # Backward pass                                              │ ║
║  │     output_error = output - y                                    │ ║
║  │     output_delta = output_error * sigmoid_derivative(output_z)   │ ║
║  │     hidden_error = output_delta @ w2.T                           │ ║
║  │     hidden_delta = hidden_error * sigmoid_derivative(hidden_z)   │ ║
║  │                                                                  │ ║
║  │     # Update weights                                             │ ║
║  │     w2 -= learning_rate * hidden_a.T @ output_delta              │ ║
║  │     b2 -= learning_rate * np.sum(output_delta, axis=0)           │ ║
║  │     w1 -= learning_rate * X.T @ hidden_delta                     │ ║
║  │     b1 -= learning_rate * np.sum(hidden_delta, axis=0)           │ ║
║  │                                                                  │ ║
║  │ Lines: ~100                                                      │ ║
║  │ Time to write: 1.5 hours                                         │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  TODAY (Keras):                                                        ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │ model = keras.Sequential([                                       │ ║
║  │     layers.Dense(4, activation='sigmoid', input_shape=(2,)),     │ ║
║  │     layers.Dense(1, activation='sigmoid')                        │ ║
║  │ ])                                                               │ ║
║  │                                                                  │ ║
║  │ model.compile(                                                   │ ║
║  │     optimizer='adam',                                            │ ║
║  │     loss='binary_crossentropy',                                  │ ║
║  │     metrics=['accuracy']                                         │ ║
║  │ )                                                                │ ║
║  │                                                                  │ ║
║  │ model.fit(X, y, epochs=1000, batch_size=4, verbose=0)            │ ║
║  │                                                                  │ ║
║  │ Lines: 10                                                        │ ║
║  │ Time to write: 2 minutes                                         │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  RESULT: SAME NETWORK, 10x LESS CODE! ✨                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
"""

print(comparison)

# MODEL SAVING & LOADING
# ============================================

print("\n" + "="*80)
print("MODEL PERSISTENCE: SAVE & LOAD")
print("="*80)

print("""
SAVING MODELS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Save entire model (architecture + weights + optimizer state)
model.save('my_model.keras')        # Recommended (TF 2.x)
model.save('my_model.h5')           # Legacy format

# Save only weights
model.save_weights('weights.h5')


LOADING MODELS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Load entire model
model = keras.models.load_model('my_model.keras')

# Load weights into existing model
model.load_weights('weights.h5')


WHY SAVE MODELS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Deployment: Load trained model in production
✅ Resume training: Continue from checkpoint
✅ Sharing: Share models with team
✅ Versioning: Track model improvements
""")

# Save the XOR model
model.save('xor_model.keras')
print("✅ Model saved: xor_model.keras")

# Load it back
loaded_model = keras.models.load_model('xor_model.keras')
print("✅ Model loaded successfully")

# Verify it works
test_pred = loaded_model.predict(X_xor, verbose=0)
print(f"✅ Loaded model predictions match: {np.allclose(predictions, test_pred)}")

# VISUALIZATION
# ============================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training History
axes[0].plot(history.history['loss'], linewidth=2, color='#e74c3c', label='Training Loss')
axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Loss (Binary Crossentropy)', fontweight='bold', fontsize=11)
axes[0].set_title('Keras Training Loss', fontweight='bold', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Plot 2: Accuracy
axes[1].plot(history.history['accuracy'], linewidth=2, color='#2ecc71', label='Training Accuracy')
axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Accuracy', fontweight='bold', fontsize=11)
axes[1].set_title('Keras Training Accuracy', fontweight='bold', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (100%)')

plt.suptitle('KERAS XOR NETWORK TRAINING', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('01_keras_training.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 01_keras_training.png")

print("\n" + "="*80)
print("SESSION 1 COMPLETE: TensorFlow/Keras Fundamentals Mastered!")
print("="*80)
print("\n☕ Take a 15-minute break before Fashion MNIST!")
