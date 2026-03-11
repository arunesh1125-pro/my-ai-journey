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

# LOGISTIC REGRESSION WORKFLOW

print("\n" + "="*70)
print("LOGISTIC REGRESSION: STEP-BY-STEP")
print("="*70)

workflow = """"
STEP 1: LINEAR COMBINATION
    z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
    (Just like linear regression!)

STEP 2: APPLY SIGMOID FUNCTION
    P(y=1) = σ(z) = 1 / (1 + e^(-z))
    This gives us probability (0 to 1)

STEP 3: MAKE PREDICTION
    If P(y=1) >= 0.5 -> Predict Class 1
    If P(y=1) < 0.5  -> Predict Class 0

STEP 4: TRAINING (Finding b₀, b₁, b₂, ...)   # Unlike MSE for Linear Regression, Here Log Loss (Binary Cross-Entropy) was used for Logistic regression
    Minimizes: Log Loss (Binary Cross-Entropy)

    Log Loss = -1/n × Σ[y·log(p) + (1-y)·log(1-p)]

    where:
    y = actual class (0 or 1)
    p = predicted probability

    Intution:
    • Penalizes confident wrong predictions heavily
    • Rewards confident correct predictions

Example Calculation:
────────────────────────────────────────────────────
Actual = 1, Predicted P = 0.9 -> Loss = -log(0.9) = 0.10 (low, good!)
Actual = 1, Predicted P = 0.1 -> Loss = -log(0.1) = 2.30 (high, bad!)
Actual = 1, Predicted P = 0.2 -> Loss = -log(0.8) = 0.22 (low, good!)
"""
print(workflow)

# SIMPLE EXAMPLE: EMAIL SPAM CLASSIFIER

print("\n" + "="*70)
print("SIMPLE EXAMPLE: EMAIL SPAM DETECTION")
print("="*70)

# Generate simple spam data
np.random.seed(99)
n_emails = 100

# Features
num_exclamation = np.concatenate([
    np.random.poisson(1, 50),       # Normal emails: few !
    np.random.poisson(8, 50)        # Spam emails: many !
])

num_links = np.concatenate([
    np.random.poisson(1, 50),        # Normal emails: few links
    np.random.poisson(6, 50)         # Spam emails: many links
])

# Labels (0 = not spam, 1 = spam)
is_spam = np.array([0]*50 + [1]*50)

# Shuffle
shuffle_idx = np.random.permutation(n_emails)
num_exclamation = num_exclamation[shuffle_idx]
num_links = num_links[shuffle_idx]
is_spam = is_spam[shuffle_idx]

# Create DataFrame
df_spam = pd.DataFrame({
    'Exclamation_Marks': num_exclamation,
    'Num_Links': num_links,
    'Is_Spam': is_spam
})

print("\nEmail Dataset (100 emails): ")
print(df_spam.head(10))
print("\nClass Distribution: ")
print(df_spam['Is_Spam'].value_counts())

# Prepare data
X = df_spam[['Exclamation_Marks', 'Num_Links']].values
y = df_spam['Is_Spam'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=42
)

# Training Logistic Regresssion
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"\n✅ Model Trained!")
print("\nLearned Coefficients:")
print(f"    Intercept (b₀): {model.intercept_[0]:.3f}")
print(f"    Exclamation_Marks (b₁): {model.coef_[0][0]:.3f}")
print(f"    Num_Links (b₂): {model.coef_[0][1]:.3f}")

print(f"Logistic Regression Equation: ")
print(f"    z = {model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f} x Exclamation + {model.coef_[0][1]:.3f} x Links")
print(f"  P(Spam) = 1 / (1 + e^(-z))")

# Make predictions
y_pred = model.predict(X_test) # predicct class labels
y_pred_proba = model.predict_proba(X_test)[:, 1] # predict prbability estimates # Probability of spam

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Acccuracy: {accuracy:.2%}")

# Show some predictions
print(f"\n{'Actual':>10} {'Predicted':>10} {'Probablity':>15} {'Decision':>15}")
for i in range(min(10, len(y_test))):
    actual = 'Spam' if y_test[i] == 1 else 'Not Spam'
    predicted = 'Spam' if y_pred[i] == 1 else 'Not Spam'
    prob = y_pred_proba[i]
    decision = '✓' if y_test[i] == y_pred[i] else '✗'
    print(f"{actual:>10} {predicted:>10} {prob:>15.2%} {decision:>15}")

# Visualize decision boundary
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Data points
spam = df_spam[df_spam['Is_Spam'] == 1]
not_spam = df_spam[df_spam['Is_Spam'] == 0]

axes[0].scatter(not_spam['Exclamation_Marks'], not_spam['Num_Links'],
                c='blue', s=80, alpha=0.6, edgecolors='black',
                label='Not Spam', marker='o')
axes[0].scatter(spam['Exclamation_Marks'], not_spam['Num_Links'],
                c='red', s=80, alpha=0.6, edgecolors='black',
                label='Spam', marker='^')
axes[0].set_xlabel('Exclamation Marks', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Number of Links', fontweight='bold', fontsize=12)
axes[0].set_title('Email Spam Classification', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)


# Decision boundary
# Create mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))    # This function creates a grid of points (xx, yy) that cover the entire plot area. This grid is used to evaluate the model's prediction at every single pixel on the graph [1].
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] # ravel - flatten mult-dimensional array into a one-dimension array
Z = Z.reshape(xx.shape)

# Plot contour
contour = axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6) # .contourf = create a filled contour plot, which colors the areas b/w contour lines to visualize 3D func to 2D plane.
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidth=3)

#Overlay data
axes[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
                c='blue', s=100, alpha=0.8, edgecolors='black',
                label='Not Spam (Test)', marker='o')
axes[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
                c='red', s=100, alpha=0.8, edgecolors='black',
                label='Spam (Test)', marker='^')
axes[1].set_xlabel('Exclamation Marks', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Number of Links', fontweight='bold', fontsize=12)
axes[1].set_title('Decision Boundary & Probabilities', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
plt.colorbar(contour, ax=axes[1], label='P(Spam)')

plt.tight_layout()
plt.savefig('03_spam_classifier.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 03_spam_classifier.png")

print("\n" + "="*70)
print("LOGISTIC REGRESSION THEORY COMPLETE!")
print("="*70)