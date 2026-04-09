"""
COMPARISON: NEURAL NETWORK VS LOGISTIC REGRESSION ON XOR
=========================================================
Proving that neural networks can solve problems that linear models cannot!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

print("="*80)
print("NEURAL NETWORK vs LOGISTIC REGRESSION: XOR SHOWDOWN")
print("="*80)

np.random.seed(42)

# XOR DATA

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

print("\nXOR Problem:")
print(f"{'Input A':<10} {'Input B':<10} {'Output (XOR)':<15}")
print("-" * 35)
for i in range(len(X)):
    print(f"{X[i, 0]:<10} {X[i, 1]:<10} {y[i]:<15}")

# LOGISTIC REGRESSION (WILL FAIL)

print("\n" + "="*80)
print("ATTEMPT 1: LOGISTIC REGRESSION")
print("="*80)

lr_model = LogisticRegression()
lr_model.fit(X, y)

lr_predictions = lr_model.predict(X)
lr_proba = lr_model.predict_proba(X)[:, 1]
lr_accuracy = np.mean(lr_predictions == y) * 100

print(f"\n📊 Logistic Regression Results:")
print(f"{'Input A':<10} {'Input B':<10} {'True':<10} {'Predicted':<12} {'Probability':<15} {'Status':<10}")
print("-" * 67)
for i in range(len(X)):
    status = "✅" if lr_predictions[i] == y[i] else "❌"
    print(f"{X[i, 0]:<10} {X[i, 1]:<10} {y[i]:<10} {lr_predictions[i]:<12} {lr_proba[i]:<15.4f} {status:<10}")

print(f"\n🎯 Logistic Regression Accuracy: {lr_accuracy:.1f}%")
print(f"💔 FAILED! Cannot learn XOR pattern!")

# NEURAL NETWORK (FROM SESSION 2)

print("\n" + "="*80)
print("ATTEMPT 2: NEURAL NETWORK")
print("="*80)

# Quick NN implementation (reusing from previous)
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Architecture
input_size = 2
hidden_size = 4
output_size = 1

# Intialize weights
weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5 
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
bias_output = np.zeros((1, output_size))

# Training parameters
learning_rate  = 0.5
epochs = 10000

# Prepare y for NN
y_nn = y.reshape(-1, 1)

print(f"Training Neural Network...")
print(f"  Architecture: {input_size} → {hidden_size} → {output_size}")
print(f"  Epochs: {epochs:,}")
print(f"  Learning Rate: {learning_rate}\n")

# Training loop (compact version)
for epoch in range(epochs):
    # Forward pass
    hidden_z = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_activation = sigmoid(hidden_z)
    output_z = np.dot(hidden_activation, weights_hidden_output) + bias_output
    output = sigmoid(output_z)

    # Backward Pass
    output_error = output - y_nn
    output_delta = output_error * sigmoid_derivative(output_z)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_z)

    # Update weights
    weights_hidden_output -= learning_rate * hidden_activation.T.dot(output_delta) / len(X)
    bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True) / len(X)
    weights_input_hidden -= learning_rate * X.T.dot(hidden_delta) / len(X)
    bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / len(X)

# Final predictions
hidden_z = np.dot(X, weights_input_hidden) + bias_hidden
hidden_activation = sigmoid(hidden_z)
output_z = np.dot(hidden_activation, weights_hidden_output) + bias_output
nn_predictions_proba = sigmoid(output_z).flatten()
nn_predictions = np.round(nn_predictions_proba).astype(int)
nn_accuracy = np.mean(nn_predictions == y) * 100

print(f"✅ Training Complete!")
print(f"\n📊 Neural Network Results:")
print(f"{'Input A':<10} {'Input B':<10} {'True':<10} {'Predicted':<12} {'Probability':<15} {'Status':<10}")
print("-" * 67)
for i in range(len(X)):
    status = "✅" if nn_predictions[i] == y[i] else "❌"
    print(f"{X[i, 0]:<10} {X[i, 1]:<10} {y[i]:<10} {nn_predictions[i]:<12} {nn_predictions_proba[i]:<15.4f} {status:<10}")

print(f"\n🎯 Neural Network Accuracy: {nn_accuracy:.1f}%")
print(f"🎉 SUCCESS! Learned XOR pattern perfectly!")

# VISUAL COMPARISON

print("\n" + "="*80)
print("CREATING VISUAL COMPARISON")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colors for plotting
colors = ['blue' if y[i]==0 else 'red' for i in range(len(y))]

# Create mesh for decision boundaries
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200), 
                     np.linspace(-0.3, 1.3, 200))
mesh_points = np.c_[xx.ravel(), yy.ravel()]

# Plot 1: Original XOR Problem
axes[0].scatter(X[:, 0], X[:, 1], c=colors, s=500, edgecolors='black', linewidth=3)
for i, (x_val, y_val) in enumerate(zip(X, y)):
    axes[0].annotate(f'({int(x_val[0])},{int(x_val[1])})→{int(y_val)}',
                    xy=(x_val[0], x_val[1]), xytext=(10, 10), 
                    textcoords='offset points', fontsize=11, fontweight='bold')

axes[0].set_xlabel('Input A', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Input B', fontweight='bold', fontsize=12)
axes[0].set_title('XOR Problem\n(Linearly Inseparable)', fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-0.3, 1.3)
axes[0].set_ylim(-0.3, 1.3)

# Legend
axes[0].plot([], [], 'o', color='blue', markersize=15, label='Output = 0')
axes[0].plot([], [], 'o', color='red', markersize=15, label='Output = 1')
axes[0].legend(fontsize=11, loc='upper right')

# Plot 2: Logistic Regression (FAILS)
lr_mesh = lr_model.predict_proba(mesh_points)[:, 1].reshape(xx.shape)
contour2 = axes[1].contourf(xx, yy, lr_mesh, levels=20, cmap='RdYlBu_r', alpha=0.7)
axes[1].scatter(X[:, 0], X[:, 1], c=colors, s=500, edgecolors='black', linewidth=3)
axes[1].contour(xx, yy, lr_mesh, levels=[0.5], colors='black', linewidths=3, linestyles='--')

axes[1].set_xlabel('Input A', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Input B', fontweight='bold', fontsize=12)
axes[1].set_title(f'Logistic Regression ❌\nAccuracy: {lr_accuracy:.1f}%', 
                 fontweight='bold', fontsize=14, color='red')
axes[1].grid(True, alpha=0.3)

# Add colorbar
cbar2 = plt.colorbar(contour2, ax=axes[1])
cbar2.set_label('Prediction', fontweight='bold')

# Plot 3: Neural Network (SUCCEEDS!)
# Get NN predictions for mesh
mesh_hidden_z = np.dot(mesh_points, weights_input_hidden) + bias_hidden
mesh_hidden_a = sigmoid(mesh_hidden_z)
mesh_output_z = np.dot(mesh_hidden_a, weights_hidden_output) + bias_output
nn_mesh = sigmoid(mesh_output_z).reshape(xx.shape)

contour3 = axes[2].contourf(xx, yy, nn_mesh, levels=20, cmap='RdYlBu_r', alpha=0.7)
axes[2].scatter(X[:, 0], X[:, 1], c=colors, s=500, edgecolors='black', linewidth=3)
axes[2].contour(xx, yy, nn_mesh, levels=[0.5], colors='black', linewidths=3, linestyles='--')

axes[2].set_xlabel('Input A', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Input B', fontweight='bold', fontsize=12)
axes[2].set_title(f'Neural Network ✅\nAccuracy: {nn_accuracy:.1f}%', 
                 fontweight='bold', fontsize=14, color='green')
axes[2].grid(True, alpha=0.3)

# Add colorbar
cbar3 = plt.colorbar(contour3, ax=axes[2])
cbar3.set_label('Prediction', fontweight='bold')

plt.suptitle('WHY NEURAL NETWORKS? XOR PROBLEM COMPARISON', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('03_nn_vs_lr_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 03_nn_vs_lr_comparison.png")

# ============================================
# SUMMARY TABLE
# ============================================

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

comparison_text = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                     MODEL COMPARISON: XOR PROBLEM                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Model               │  Accuracy  │  Can Solve XOR?  │  Complexity   ║
║  ───────────────────┼────────────┼──────────────────┼──────────────  ║
║  Logistic Regression │   {lr_accuracy:.1f}%     │       ❌ NO      │   Simple      ║
║  Neural Network      │  {nn_accuracy:.1f}%     │       ✅ YES     │   Moderate    ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  WHY THE DIFFERENCE?                                                  ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                                       ║
║  Logistic Regression:                                                 ║
║  • Learns LINEAR decision boundary (straight line)                   ║
║  • Cannot separate XOR pattern (needs curved boundary)               ║
║  • Best it can do: classify everything as 0 or 1 (50% accuracy)     ║
║                                                                       ║
║  Neural Network:                                                      ║
║  • Learns NON-LINEAR decision boundary (curved)                      ║
║  • Hidden layer creates intermediate features                        ║
║  • Combines simple patterns to form complex decisions                ║
║  • Achieves 100% accuracy!                                           ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  KEY INSIGHT:                                                         ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                                       ║
║  Neural networks are "universal approximators"                        ║
║  → Can learn ANY function given enough neurons!                       ║
║                                                                       ║
║  This is why deep learning works for:                                 ║
║  • Image recognition (millions of pixels)                            ║
║  • Speech recognition (complex audio patterns)                       ║
║  • Language understanding (semantic relationships)                   ║
║  • Game playing (strategic decision making)                          ║
║                                                                       ║
║  If XOR (4 data points) needs a neural network,                      ║
║  imagine what's needed for real-world problems! 🚀                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

print(comparison_text)

# ============================================
# WHEN TO USE WHAT?
# ============================================

print("\n" + "="*80)
print("PRACTICAL DECISION GUIDE")
print("="*80)

decision_guide = """
┌────────────────────┬──────────────────────┬──────────────────────┐
│ Use Case           │ Logistic Regression  │ Neural Network       │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Linear patterns    │       ✅ BEST        │    ✅ Works          │
│ Non-linear patterns│       ❌ FAILS       │    ✅ BEST           │
│ Small dataset      │       ✅ BEST        │    ⚠️ May overfit   │
│ Large dataset      │       ✅ Good        │    ✅ BEST           │
│ Interpretability   │       ✅ High        │    ❌ Black box      │
│ Training speed     │       ✅ Fast        │    ⚠️ Slower         │
│ Images/Audio/Text  │       ❌ Poor        │    ✅ BEST           │
│ Tabular data       │       ✅ Often good  │    ✅ Good           │
└────────────────────┴──────────────────────┴──────────────────────┘

RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Start with Logistic Regression (simple, fast, interpretable)
✅ If accuracy is not good → Try Neural Network
✅ For images/audio/text → Go straight to Neural Network (or CNN/RNN)
✅ For structured/tabular → Try both, pick best

Rule of thumb:
- Simple problem + small data = Logistic Regression
- Complex patterns + lots of data = Neural Network
"""

print(decision_guide)

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)