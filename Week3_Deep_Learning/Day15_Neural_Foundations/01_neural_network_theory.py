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