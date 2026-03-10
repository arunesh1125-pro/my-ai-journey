"""
LOGISTIC REGRESSION: MATHEMATICAL FOUNDATION
============================================
How Logistic Regression works under the hood
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("="*70)
print("LOGISTIC REGRESSION: THEORY & MATHEMATICS")
print("="*70)

# WHAT IS LOGISTIC REGRESSION?

print("""
╔══════════════════════════════════════════════════════════════╗
║              WHY LOGISTIC REGRESSION?                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Problem with Linear Regression for Classification:         ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ Linear: y = mx + b                                     │ ║
║  │ Output: Any number (-∞ to +∞)                          │ ║
║  │                                                        │ ║
║  │ But we need:                                           │ ║
║  │ • Probabilities (0 to 1)                               │ ║
║  │ • Binary predictions (0 or 1)                          │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  Solution: Logistic (Sigmoid) Function                      ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │                    1                                   │ ║
║  │ σ(z) = ─────────────────                              │ ║
║  │          1 + e^(-z)                                    │ ║
║  │                                                        │ ║
║  │ Where: z = b₀ + b₁x₁ + b₂x₂ + ... (linear combo)      │ ║
║  │                                                        │ ║
║  │ Properties:                                            │ ║
║  │ • Output always between 0 and 1                        │ ║
║  │ • S-shaped curve                                       │ ║
║  │ • σ(0) = 0.5 (middle point)                           │ ║
║  │ • As z → +∞, σ(z) → 1                                 │ ║
║  │ • As z → -∞, σ(z) → 0                                 │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================
# VISUALIZE SIGMOID FUNCTION
# ============================================

print("\n" + "="*70)
print("THE SIGMOID (LOGISTIC) FUNCTION")
print("="*70)

def sigmoid(z):
    """Sigmoid function: maps any real number to (0, 1)"""
    return 1 / (1 + np.exp(-z))

# Generate z values
z = np.linspace(-10, 10, 200)
prob = sigmoid(z)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Sigmoid function
axes[0].plot(z, prob, linewidth=3, color='#2ecc71', label='Sigmoid: σ(z)')
axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold (0.5)')
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[0].axhline(y=0, color='black', linewidth=1)
axes[0].axhline(y=1, color='black', linewidth=1)
axes[0].fill_between(z, 0, prob, where=(prob >= 0.5), alpha=0.2, color='red', label='Predict: Class 1')
axes[0].fill_between(z, 0, prob, where=(prob < 0.5), alpha=0.2, color='blue', label='Predict: Class 0')
axes[0].set_xlabel('z (linear combination)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('P(Class = 1)', fontweight='bold', fontsize=12)
axes[0].set_title('Sigmoid Function', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.1, 1.1)

# Add annotations
axes[0].annotate('z=0 → P=0.5', xy=(0, 0.5), xytext=(2, 0.7),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=11, fontweight='bold')
axes[0].annotate('z→+∞ → P→1', xy=(7, 0.95), xytext=(4, 0.85),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2),
                 fontsize=11, fontweight='bold')
axes[0].annotate('z→-∞ → P→0', xy=(-7, 0.05), xytext=(-4, 0.15),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                 fontsize=11, fontweight='bold')

# Plot 2: Compare Linear vs Sigmoid
x_linear = np.linspace(-5, 5, 100)
y_linear = 0.2 * x_linear + 0.5  # Linear function

axes[1].plot(x_linear, y_linear, 'r--', linewidth=2, label='Linear Regression', alpha=0.7)
axes[1].plot(z, prob, 'g-', linewidth=3, label='Logistic Regression (Sigmoid)')
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].axhline(y=1, color='black', linewidth=1)
axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Input (x)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Output', fontweight='bold', fontsize=12)
axes[1].set_title('Linear vs Logistic Regression', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.5, 1.5)

# Highlight problems with linear
axes[1].fill_between([-5, 5], [1.5, 1.5], [1, 1], alpha=0.3, color='red', 
                      label='Invalid probabilities (>1)')
axes[1].fill_between([-5, 5], [0, 0], [-0.5, -0.5], alpha=0.3, color='red')
axes[1].text(-3, 1.2, '❌ Invalid\n(>1)', fontsize=10, fontweight='bold', color='red')
axes[1].text(-3, -0.3, '❌ Invalid\n(<0)', fontsize=10, fontweight='bold', color='red')
axes[1].text(2, 0.7, '✅ Valid\n(0 to 1)', fontsize=10, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('02_sigmoid_function.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 02_sigmoid_function.png")