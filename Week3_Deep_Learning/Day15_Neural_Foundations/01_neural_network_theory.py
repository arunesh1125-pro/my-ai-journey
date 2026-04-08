"""
NEURAL NETWORKS: FROM BIOLOGICAL TO ARTIFICIAL
===============================================
Understanding the fundamentals before we build
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("NEURAL NETWORK FOUNDATIONS")
print("="*80)

# WHY NEURAL NETWORKS?

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                 WHY NEURAL NETWORKS EXIST                              ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  THE PROBLEM WITH CLASSICAL ML:                                       ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Classical ML (Week 2):                                          │ ║
║  │  • Linear Regression: Draws straight line                        │ ║
║  │  • Logistic Regression: Draws straight boundary                  │ ║
║  │  • Decision Trees: Rectangular boundaries                        │ ║
║  │                                                                  │ ║
║  │  Example Problem: XOR                                            │ ║
║  │                                                                  │ ║
║  │      Input A  │  Input B  │  Output (A XOR B)                   │ ║
║  │      ────────┼───────────┼──────────────────                    │ ║
║  │         0    │     0     │        0                             │ ║
║  │         0    │     1     │        1                             │ ║
║  │         1    │     0     │        1                             │ ║
║  │         1    │     1     │        0                             │ ║
║  │                                                                  │ ║
║  │  Plot this on a graph:                                           │ ║
║  │                                                                  │ ║
║  │      Input B                                                     │ ║
║  │        1  │  🔴      🔵                                          │ ║
║  │           │                                                      │ ║
║  │        0  │  🔵      🔴                                          │ ║
║  │           └───────────── Input A                                │ ║
║  │              0        1                                          │ ║
║  │                                                                  │ ║
║  │  🔴 = Output 1   🔵 = Output 0                                  │ ║
║  │                                                                  │ ║
║  │  CAN'T SEPARATE WITH A STRAIGHT LINE! ❌                        │ ║
║  │                                                                  │ ║
║  │  Logistic Regression FAILS on XOR problem!                       │ ║
║  │  This is called "linearly inseparable"                           │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  THE SOLUTION: NEURAL NETWORKS                                         ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Neural Networks can learn NON-LINEAR patterns!                  │ ║
║  │                                                                  │ ║
║  │  They can draw CURVED decision boundaries                        │ ║
║  │  They can solve XOR! ✅                                          │ ║
║  │                                                                  │ ║
║  │  How? By combining multiple simple functions                     │ ║
║  │  in clever ways through LAYERS                                   │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# BIOLOGICAL INSPIRATION

print("\n" + "="*80)
print("FROM BRAIN TO COMPUTER")
print("="*80)

print("""
BIOLOGICAL NEURON (Brain):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Dendrites ──→  Cell Body  ──→  Axon  ──→  Synapse
    (Inputs)       (Process)       (Output)   (Connections)

How it works:
1. Dendrites receive signals from other neurons
2. Cell body SUMS all incoming signals
3. If sum > threshold → Neuron FIRES (sends signal)
4. Signal travels down axon to other neurons


ARTIFICIAL NEURON (Computer):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Inputs (x₁, x₂, x₃)
       │
       ├──→ Weights (w₁, w₂, w₃)
       │
       ├──→ WEIGHTED SUM: z = (x₁×w₁ + x₂×w₂ + x₃×w₃) + bias
       │
       └──→ ACTIVATION FUNCTION: a = f(z)
              │
              └──→ Output


Mathematical Formula:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Weighted Sum
   z = w₁·x₁ + w₂·x₂ + w₃·x₃ + b
   
   Where:
   • x = inputs (data)
   • w = weights (learned parameters)
   • b = bias (learned parameter)

Step 2: Activation Function
   a = f(z)
   
   Example: Sigmoid
   a = 1 / (1 + e^(-z))
   
   Output: Number between 0 and 1


Example Calculation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Inputs:  x₁=0.5, x₂=0.8, x₃=0.2
Weights: w₁=0.4, w₂=-0.3, w₃=0.7
Bias:    b=0.1

Step 1: Weighted Sum
z = (0.5×0.4) + (0.8×-0.3) + (0.2×0.7) + 0.1
z = 0.2 + (-0.24) + 0.14 + 0.1
z = 0.2

Step 2: Activation (Sigmoid)
a = 1 / (1 + e^(-0.2))
a = 1 / (1 + 0.8187)
a = 1 / 1.8187
a = 0.550 ✅

Output: 0.550 (55% probability)
""")

# Visual demonstration
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

# ACTIVATION FUNCTIONS


print("\n" + "="*80)
print("ACTIVATION FUNCTIONS: THE NON-LINEARITY SECRET")
print("="*80)

print("""
WHY DO WE NEED ACTIVATION FUNCTIONS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without activation functions:
  Layer 1: z₁ = w₁·x + b₁
  Layer 2: z₂ = w₂·z₁ + b₂
  Layer 2: z₂ = w₂·(w₁·x + b₁) + b₂
         = (w₂·w₁)·x + (w₂·b₁ + b₂)
         = W·x + B   ← STILL JUST A STRAIGHT LINE!

This is just linear regression with extra steps! ❌

With activation functions:
  Layer 1: a₁ = sigmoid(w₁·x + b₁)  ← CURVED!
  Layer 2: a₂ = sigmoid(w₂·a₁ + b₂)  ← MORE CURVES!
  
Now we can learn COMPLEX patterns! ✅


COMMON ACTIVATION FUNCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SIGMOID (Logistic)
   Formula: σ(z) = 1 / (1 + e^(-z))
   Range: 0 to 1
   Shape: S-curve
   
   Use: Output layer for binary classification
   Pros: Smooth, probabilistic output
   Cons: Vanishing gradients (problem for deep networks)


2. ReLU (Rectified Linear Unit) ⭐ MOST POPULAR!
   Formula: f(z) = max(0, z)
   Range: 0 to +∞
   Shape: Bent line (0 if z<0, linear if z>0)
   
   Use: Hidden layers (most modern networks)
   Pros: Fast, avoids vanishing gradients
   Cons: "Dying ReLU" problem (neurons can die)


3. TANH (Hyperbolic Tangent)
   Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
   Range: -1 to 1
   Shape: S-curve (like sigmoid but centered at 0)
   
   Use: Hidden layers (less common now)
   Pros: Zero-centered (better than sigmoid)
   Cons: Still has vanishing gradient


4. SOFTMAX (for multi-class)
   Formula: σ(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
   Range: 0 to 1 (sums to 1 across all outputs)
   
   Use: Output layer for multi-class classification
   Pros: Converts scores to probabilities
   Example: [2.3, 1.1, 0.5] → [0.70, 0.21, 0.09]
""")

# Visualize activation functions
z = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ACTIVATION FUNCTIONS COMPARISON', fontsize=16, fontweight='bold')

# Sigmoid
axes[0, 0].plot(z, sigmoid(z), 'b-', linewidth=3)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('SIGMOID: σ(z) = 1/(1+e^(-z))', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Input (z)', fontweight='bold')
axes[0, 0].set_ylabel('Output', fontweight='bold')
axes[0, 0].text(-4, 0.9, 'Range: [0, 1]', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
axes[0, 0].text(-4, 0.7, 'Use: Binary output', fontsize=9)

# ReLU
axes[0, 1].plot(z, relu(z), 'r-', linewidth=3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('ReLU: f(z) = max(0, z)', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Input (z)', fontweight='bold')
axes[0, 1].set_ylabel('Output', fontweight='bold')
axes[0, 1].text(-4, 4, 'Range: [0, ∞)', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
axes[0, 1].text(-4, 3, 'Use: Hidden layers ⭐', fontsize=9)

# Tanh
axes[1, 0].plot(z, tanh(z), 'g-', linewidth=3)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('TANH: tanh(z)', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Input (z)', fontweight='bold')
axes[1, 0].set_ylabel('Output', fontweight='bold')
axes[1, 0].text(-4, 0.8, 'Range: [-1, 1]', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
axes[1, 0].text(-4, 0.5, 'Use: Hidden layers', fontsize=9)

# Step function (for comparison)
step = np.where(z >= 0, 1, 0)
axes[1, 1].plot(z, step, 'm-', linewidth=3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('STEP FUNCTION (OLD)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Input (z)', fontweight='bold')
axes[1, 1].set_ylabel('Output', fontweight='bold')
axes[1, 1].text(-4, 0.9, 'Range: {0, 1}', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
axes[1, 1].text(-4, 0.7, 'Use: Historical only', fontsize=9)

plt.tight_layout()
plt.savefig('01_activation_functions.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_activation_functions.png")

# NEURAL NETWORK ARCHITECTURE

print("\n" + "="*80)
print("NEURAL NETWORK ARCHITECTURE")
print("="*80)

print("""
ANATOMY OF A NEURAL NETWORK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Input Layer   →   Hidden Layer 1   →   Hidden Layer 2   →   Output Layer
    
      x₁ ●               ●                      ●                    ● ŷ₁
           ╲           ╱   ╲                  ╱   ╲                ╱
      x₂ ● ─────●         ●────────●         ●─────● ŷ₂
           ╱           ╲   ╱                  ╲   ╱                ╲
      x₃ ●               ●                      ●                    ● ŷ₃
    
    (3 neurons)    (4 neurons)           (4 neurons)          (3 neurons)


LAYER TYPES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. INPUT LAYER
   • First layer
   • One neuron per feature
   • No activation function
   • Just passes data forward
   • Example: 3 neurons for [age, income, credit_score]

2. HIDDEN LAYERS
   • Middle layers (can have many!)
   • Learn complex features
   • Use ReLU activation (usually)
   • Each neuron connects to ALL neurons in previous layer
   • Example: Layer 1 learns simple patterns, Layer 2 combines them

3. OUTPUT LAYER
   • Last layer
   • Size depends on task:
     - Binary classification: 1 neuron (Sigmoid)
     - Multi-class (3 classes): 3 neurons (Softmax)
     - Regression: 1 neuron (Linear, no activation)


FORWARD PROPAGATION (How data flows):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input → Hidden Layer 1 → Hidden Layer 2 → Output → Prediction

Example with numbers:

Input: x = [0.5, 0.8]

Layer 1 (2 neurons):
  Neuron 1: z₁ = (0.5×0.4) + (0.8×-0.2) + 0.1 = 0.14
           a₁ = ReLU(0.14) = 0.14
           
  Neuron 2: z₂ = (0.5×0.6) + (0.8×0.3) + (-0.1) = 0.44
           a₂ = ReLU(0.44) = 0.44
  
  Output: [0.14, 0.44]

Layer 2 (1 neuron - output):
  z₃ = (0.14×0.5) + (0.44×0.8) + 0.0 = 0.422
  a₃ = Sigmoid(0.422) = 0.604
  
  Final Prediction: 0.604 (60.4% probability) ✅


PARAMETERS TO LEARN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Network: Input(2) → Hidden(4) → Output(1)

Parameters:
- Layer 1: (2×4) weights + 4 biases = 12 parameters
- Layer 2: (4×1) weights + 1 bias = 5 parameters
- Total: 17 parameters to learn!

Small network, but already 17 numbers to optimize!
Large networks can have MILLIONS of parameters!
""")

# LOSS FUNCTION

print("\n" + "="*80)
print("LOSS FUNCTIONS: HOW WRONG ARE WE?")
print("="*80)

print("""
WHAT IS A LOSS FUNCTION?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loss = measure of how bad our predictions are

Goal: MINIMIZE loss (make predictions better!)


COMMON LOSS FUNCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MEAN SQUARED ERROR (MSE) - For Regression
   
   Formula: MSE = (1/n) Σ (y_true - y_pred)²
   
   Example:
   True values:     [100, 200, 150]
   Predictions:     [110, 190, 140]
   Errors:          [-10, 10, 10]
   Squared errors:  [100, 100, 100]
   MSE:             (100+100+100)/3 = 100
   
   Intuition: Penalizes large errors more (squaring!)


2. BINARY CROSS-ENTROPY - For Binary Classification
   
   Formula: BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
   
   Example:
   True label: 1 (positive class)
   Prediction: 0.8 (80% confident it's positive)
   
   BCE = -[1·log(0.8) + 0·log(0.2)]
       = -log(0.8)
       = 0.097  ← Small loss (good prediction!)
   
   If prediction was 0.2:
   BCE = -log(0.2) = 0.699  ← Large loss (bad prediction!)
   
   Intuition: Heavily penalizes confident wrong predictions


3. CATEGORICAL CROSS-ENTROPY - For Multi-class
   
   Formula: CCE = -Σ y_true · log(y_pred)
   
   Example (3 classes):
   True:        [0, 1, 0]  ← Class 2 is correct
   Prediction:  [0.1, 0.7, 0.2]
   
   CCE = -[0·log(0.1) + 1·log(0.7) + 0·log(0.2)]
       = -log(0.7)
       = 0.155
   
   Intuition: Only cares about probability assigned to true class
""")

# BACKPROPAGATION INTUITION

print("\n" + "="*80)
print("BACKPROPAGATION: THE LEARNING ALGORITHM")
print("="*80)

print("""
HOW NEURAL NETWORKS LEARN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: FORWARD PASS
   • Feed input through network
   • Calculate prediction
   • Calculate loss (how wrong we are)

STEP 2: BACKWARD PASS (Backpropagation)
   • Calculate: "How much did each weight contribute to the error?"
   • Use chain rule from calculus (don't worry about math!)
   • Go backwards from output to input

STEP 3: UPDATE WEIGHTS
   • Adjust weights to reduce loss
   • Use gradient descent: weight_new = weight_old - learning_rate × gradient


INTUITION (No Math Required!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Imagine you're hiking in fog trying to reach the valley (minimum loss):

1. Where am I? (Forward pass - current loss)
2. Which direction is downhill? (Backprop - gradients)
3. Take a small step downhill (Update weights)
4. Repeat until you reach the valley!

Learning rate = step size
- Too large: You might overshoot the valley
- Too small: Takes forever to reach valley
- Just right: Smooth descent ✅


EXAMPLE (Simplified):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial weight: w = 0.5
True output: y = 1
Learning rate: α = 0.1

Iteration 1:
  Forward: prediction = 0.6
  Loss: (1 - 0.6)² = 0.16
  Gradient: -0.8  (slope of loss curve)
  Update: w = 0.5 - 0.1×(-0.8) = 0.58

Iteration 2:
  Forward: prediction = 0.68
  Loss: (1 - 0.68)² = 0.10  ← BETTER!
  Gradient: -0.64
  Update: w = 0.58 - 0.1×(-0.64) = 0.644

...keep going until loss is tiny!


KEY INSIGHTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Neural networks learn by trial and error
✅ Backpropagation tells us how to adjust each weight
✅ We do this thousands of times (epochs)
✅ Eventually, network learns the pattern!

You DON'T need to understand the calculus!
Frameworks (TensorFlow, PyTorch) do it automatically!
""")

print("\n" + "="*80)
print("THEORY COMPLETE!")
print("="*80)
print("\n🎓 Key Takeaways:")
print("  1. Neural networks solve problems linear models can't (XOR!)")
print("  2. Activation functions add non-linearity (curves!)")
print("  3. Forward pass = make prediction")
print("  4. Backpropagation = learn from mistakes")
print("  5. Loss function = how wrong we are")
print("\n☕ Take a 15-minute break!")
print("   Next: Build XOR solver from scratch!")

# WHAT DID WE LEARN ?

print("\n" + "="*80)
print("WHAT DID WE JUST DO?")
print("="*80)

print("""
🎓 KEY ACHIEVEMENTS:

1. ✅ Built a neural network FROM SCRATCH (no frameworks!)
2. ✅ Solved XOR problem (impossible for linear models)
3. ✅ Implemented forward propagation (making predictions)
4. ✅ Implemented backpropagation (learning from mistakes)
5. ✅ Trained for 10,000 epochs (got loss from {:.4f} → {:.6f})
6. ✅ Achieved 100% accuracy on XOR!

💡 DEEP INSIGHTS:

- Neural networks learn by ADJUSTING WEIGHTS
- Backpropagation tells us HOW to adjust them
- More hidden neurons = more capacity to learn complex patterns
- But too many = overfitting! (we'll learn about this)

🔍 WHAT HAPPENED INSIDE:

The hidden layer learned to create INTERMEDIATE REPRESENTATIONS:
- Some neurons detect "input A is 1"
- Some neurons detect "input B is 1"
- Output layer COMBINES these to compute XOR!

This is the MAGIC of neural networks:
→ They automatically learn useful features from data! ✨

🎯 WHY THIS MATTERS:

XOR is simple, but the SAME PRINCIPLE works for:
- Image recognition (millions of pixels!)
- Speech recognition (complex audio patterns)
- Language translation (semantic understanding)
- Game playing (strategic decision-making)

If we can solve XOR with 2 inputs, we can solve ANYTHING
with enough neurons, layers, and training data!""")

print("="*80)
print("SESSION 2 COMPLETE: XOR SOLVER BUILT! 🎉")
print("="*80)