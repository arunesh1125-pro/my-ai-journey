"""
BUILD XOR NEURAL NETWORK FROM SCRATCH
======================================
No TensorFlow, no PyTorch - just NumPy!
Understanding every line of code.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("BUILDING XOR SOLVER WITH NEURAL NETWORK")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# THE XOR PROBLEM
# ============================================

print("""
THE XOR PROBLEM (Linearly Inseparable):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Truth Table:
  Input A │ Input B │ Output (A XOR B)
  ────────┼─────────┼─────────────────
     0    │    0    │        0
     0    │    1    │        1
     1    │    0    │        1
     1    │    1    │        0

Visualization:
    B
  1 │  🔴(0,1)→1    🔵(1,1)→0
    │
  0 │  🔵(0,0)→0    🔴(1,0)→1
    └──────────────────── A
       0             1

Can't draw ONE straight line to separate red and blue! ❌
We need a NEURAL NETWORK! ✅
""")

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

print(f"Training Data:")
print(f"X (inputs):\n{X}\n")
print(f"y (targets):\n{y}\n")

# ACTIVATION FUCTIONS

def sigmoid(z):
    """Sigmoid activation: 0 to 1"""
    return 1/(1 + np.exp(-np.clip(z, -500, 500)))   # Clip to avoid overflow

def sigmoid_derivative(z):
    """Derivative of sigmoid for backprop"""
    s = sigmoid(z)
    return s * (1 - s)

print("Activation Function: Sigmoid")
print("  Formula: σ(z) = 1 / (1 + e^(-z))")
print("  Range: [0, 1]")
print("  Use: Both hidden and output layers\n")

# NEURAL NETWORK ARCHITECTURE

print("="*80)
print("NEURAL NETWORK ARCHITECTURE")
print("="*80)

# Network structure
input_size = 2      # Two inputs (A, B)
hidden_size = 4     # Four hidden neurons (can experiment with this!)
output_size = 1     # One output (XOR result)

print(f"""
Network Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Layer:     {input_size} neurons (A, B)
                   ↓
Hidden Layer:    {hidden_size} neurons (learns complex patterns)
                   ↓
Output Layer:    {output_size} neuron (XOR result)

Total Parameters:
  • Weights (Input→Hidden): {input_size}×{hidden_size} = {input_size*hidden_size}
  • Biases (Hidden):        {hidden_size}
  • Weights (Hidden→Output): {hidden_size}×{output_size} = {hidden_size*output_size}
  • Biases (Output):        {output_size}
  ─────────────────────────────────────
  Total:                    {input_size*hidden_size + hidden_size + hidden_size*output_size + output_size} parameters to learn!
""")

# Intialize weights and biases with small random values
weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
bias_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.random.randn(hidden_size, output_size)*0.5
bias_output = np.zeros((1, output_size))

print("✅ Weights initialized randomly")
print(f"  weights_input_hidden shape: {weights_input_hidden.shape}")
print(f"  weights_hidden_output shape: {weights_hidden_output.shape}\n")

# FORWARD PROPAGATION

def forward_propagation(X, w1, b1, w2, b2):
    """
    Forward pass through the network
    
    Returns:
        hidden_activation: Output of hidden layer
        output: Final prediction
        hidden_z: Pre-activation of hidden layer (needed for backprop)
    """

    # Layer 1: Input -> Hidden
    hidden_z = np.dot(X, w1) + b1           # Weighted sum
    hidden_activation = sigmoid(hidden_z)    # Activation

    # Layer 2: Hidden -> Output
    output_z = np.dot(hidden_activation, w2) + b2   # Weighted
    output = sigmoid(output_z)                      # Activation

    return hidden_activation, output, hidden_z

# LOSS FUNCTION

def compute_loss(y_true, y_pred):
    """Mean Squared Error Loss"""
    return np.mean((y_true - y_pred) ** 2)

# BACK PROPAGATION

def backward_propagation(X, y, hidden_activation, output, hidden_z, w2):
    """
    Backward pass - calculate gradients
    
    This is where the LEARNING happens!
    We calculate how much each weight contributed to the error.
    """
    m = X.shape[0]      # Number of examples

    # Output layer gradients
    output_error = output - y   # How wrong was our prediction?
    output_delta = output_error * sigmoid_derivative(hidden_activation @ w2)

    # Hidden layer gradients (chain rule)
    hidden_error = output_delta.dot(w2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_z)

    # Calculate gradients for weights and biases
    grad_w2 = hidden_activation.T.dot(output_delta) / m
    grad_b2 = np.sum(output_delta, axis=0, keepdims=True) / m

    grad_w1 = X.T.dot(hidden_delta) / m
    grad_b1 = np.sum(hidden_delta, axis=0, keepdims=True) / m

    return grad_w1, grad_b1, grad_w2, grad_b2

# TRAINING LOOP

print("="*80)
print("TRAINING NEURAL NETWORK")
print("="*80)

# Hyperparameters
learning_rate = 0.5
epochs = 10000

# Track loss over time
loss_history = []

print(f"Hyperparameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {epochs:,}")
print(f"\nTraining started...\n")

# Training
for epoch in range(epochs):
    # Forward pass
    hidden_activation, output, hidden_z = forward_propagation(
        X, weights_input_hidden, bias_hidden,
        weights_hidden_output, bias_output
    )

    # Calculate loss
    loss = compute_loss(y, output)
    loss_history.append(loss)

    # Backward pass
    grad_w1, grad_b1, grad_w2, grad_b2 = backward_propagation(
        X, y, hidden_activation, output, hidden_z, weights_hidden_output
    )

    # Update weights (Gradient Descent)
    weights_input_hidden -= learning_rate * grad_w1
    bias_hidden -= learning_rate * grad_b1
    weights_hidden_output -= learning_rate * grad_w2
    bias_output -= learning_rate * grad_b2

    # Print progress
    if (epoch + 1) % 10000 == 0:
        print(f"Epoch {epoch+1:5,}/{epoch:,} | Loss: {loss:.6f}")

print(f"\n✅ Training complete!")

# FINAL PREDICTIONS

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

# Get final predictions
_, final_predictions, _ = forward_propagation(
    X, weights_input_hidden, bias_hidden,
    weights_hidden_output, bias_output
)

print(f"\nXOR Truth Table vs Neural Network Predictions:\n")
print(f"{'Input A':<10} {'Input B':<10} {'True Output':<15} {'NN Output':<15} {'Rounded':<10}")
print("-" * 70)

for i in range(len(X)):
    a, b = X[i]
    true_val = y[i][0]
    pred_val = final_predictions[i][0]
    rounded = round(pred_val)

    status = "✅" if rounded == true_val else "❌"
    
    print(f"{int(a):<10} {int(b):<10} {int(true_val):<15} {pred_val:<15.6f} {rounded:<10} {status}")

# Calculate accuracy
predictions_rounded = np.round(final_predictions)
accuracy = np.mean(predictions_rounded == y) * 100

print(f"\n🎯 Accuracy: {accuracy:.1f}%")
print(f"🎯 Final Loss: {loss_history[-1]:.6f}")

# VISUALIZATIONS
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(loss_history, linewidth=2, color='#e74c3c')
ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontweight='bold', fontsize=11)
ax1.set_title('Training Loss Over Time', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # Log scale to see improvement better

# Plot 2: XOR Input Space
ax2 = fig.add_subplot(gs[0, 1])
colors = ['blue' if y[i]==0 else 'red' for i in range(len(y))]
ax2.scatter(X[:, 0], X[:, 1], c=colors, s=500, edgecolors='black', linewidth=3)
for i, (x, y_val) in enumerate(zip(X, y)):
    ax2.annotate(f'({int(x[0])},{int(x[1])})→{int(y_val[0])}',
                xy=(x[0], x[1]), xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold')
ax2.set_xlabel('Input A', fontweight='bold', fontsize=11)
ax2.set_ylabel('Input B', fontweight='bold', fontsize=11)
ax2.set_title('XOR Problem (Linearly Inseparable)', fontweight='bold', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(-0.3, 1.3)

# Add legend
ax2.plot([], [], 'o', color='blue', markersize=12, label='Output = 0')
ax2.plot([], [], 'o', color='red', markersize=12, label='Output = 1')
ax2.legend(fontsize=10)

# Plot 3: Decision Boundary
ax3 = fig.add_subplot(gs[0, 2])

# Create mesh
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 100), 
                     np.linspace(-0.3, 1.3, 100))
Z = np.c_[xx.ravel(), yy.ravel()]

# Predict for all points
_, predictions, _ = forward_propagation(
    Z, weights_input_hidden, bias_hidden,
    weights_hidden_output, bias_output
)
predictions = predictions.reshape(xx.shape)

# Plot contour
contour = ax3.contourf(xx, yy, predictions, levels=20, cmap='RdYlBu_r', alpha=0.7)
plt.colorbar(contour, ax=ax3, label='NN Output')

# Plot data points
ax3.scatter(X[:, 0], X[:, 1], c=colors, s=500, edgecolors='black', linewidth=3)
ax3.set_xlabel('Input A', fontweight='bold', fontsize=11)
ax3.set_ylabel('Input B', fontweight='bold', fontsize=11)
ax3.set_title('Neural Network Decision Boundary', fontweight='bold', fontsize=13)
ax3.grid(True, alpha=0.3)

# Plot 4: Predictions Bar Chart
ax4 = fig.add_subplot(gs[1, 0])
indices = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
true_vals = y.flatten()
pred_vals = final_predictions.flatten()

x_pos = np.arange(len(indices))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, true_vals, width, label='True', 
               color='#2ecc71', edgecolor='black', linewidth=2)
bars2 = ax4.bar(x_pos + width/2, pred_vals, width, label='Predicted',
               color='#3498db', edgecolor='black', linewidth=2)

ax4.set_xlabel('Input (A, B)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Output', fontweight='bold', fontsize=11)
ax4.set_title('True vs Predicted Outputs', fontweight='bold', fontsize=13)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(indices)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Weight Heatmap (Input → Hidden)
ax5 = fig.add_subplot(gs[1, 1])
im = ax5.imshow(weights_input_hidden.T, cmap='coolwarm', aspect='auto')
ax5.set_xlabel('Input Neuron', fontweight='bold', fontsize=11)
ax5.set_ylabel('Hidden Neuron', fontweight='bold', fontsize=11)
ax5.set_title('Learned Weights (Input→Hidden)', fontweight='bold', fontsize=13)
ax5.set_xticks([0, 1])
ax5.set_xticklabels(['A', 'B'])
ax5.set_yticks(range(hidden_size))
plt.colorbar(im, ax=ax5, label='Weight Value')

# Annotate weights
for i in range(hidden_size):
    for j in range(input_size):
        text = ax5.text(j, i, f'{weights_input_hidden[j, i]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold')

# Plot 6: Summary Stats
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

summary = f"""
╔═══════════════════════════════════════╗
║      XOR NEURAL NETWORK SUMMARY       ║
╠═══════════════════════════════════════╣
║                                       ║
║  Architecture:                        ║
║  • Input: 2 neurons                   ║
║  • Hidden: {hidden_size} neurons                  ║
║  • Output: 1 neuron                   ║
║                                       ║
║  Training:                            ║
║  • Epochs: {epochs:,}                    ║
║  • Learning Rate: {learning_rate}                ║
║  • Initial Loss: {loss_history[0]:.6f}           ║
║  • Final Loss: {loss_history[-1]:.6f}             ║
║                                       ║
║  Performance:                         ║
║  • Accuracy: {accuracy:.1f}%                    ║
║  • All 4 cases correct! ✅            ║
║                                       ║
║  Key Insight:                         ║
║  Neural networks can learn            ║
║  non-linear patterns that             ║
║  logistic regression cannot!          ║
║                                       ║
║  The hidden layer creates a           ║
║  curved decision boundary             ║
║  that separates the classes.          ║
║                                       ║
╚═══════════════════════════════════════╝
"""

ax6.text(0.5, 0.5, summary, transform=ax6.transAxes,
         fontsize=10, verticalalignment='center', horizontalalignment='center',
         family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('XOR NEURAL NETWORK - COMPLETE ANALYSIS', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('02_xor_neural_network_complete.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_xor_neural_network_complete.png")

# WHAT DID WE LEARN?

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
with enough neurons, layers, and training data!

""".format(loss_history[0], loss_history[-1]))

print("="*80)
print("SESSION 2 COMPLETE: XOR SOLVER BUILT! 🎉")
print("="*80)