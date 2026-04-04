"""
DECISION TREES: QUICK CONCEPTUAL UNDERSTANDING
==============================================
Learn decision trees efficiently - concept + practice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification

print("="*70)
print("DECISION TREES: CONCEPTUAL FOUNDATION")
print("="*70)


# WHAT ARE DECISION TREES?

print("""
╔══════════════════════════════════════════════════════════════╗
║                    WHAT ARE DECISION TREES?                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Think of it like a FLOWCHART for making decisions:         ║
║                                                              ║
║           Is Income > ₹50,000?                              ║
║          /                    \                              ║
║       YES                      NO                            ║
║        |                        |                            ║
║   Age > 30?              Has Collateral?                     ║
║    /    \                  /        \                        ║
║  YES    NO              YES         NO                       ║
║   |      |               |           |                       ║
║ APPROVE REJECT        APPROVE     REJECT                     ║
║                                                              ║
║  Key Idea: Ask Yes/No questions until you reach decision    ║
║                                                              ║
║  Real-World Examples:                                        ║
║  • Loan approval (approve or reject?)                       ║
║  • Medical diagnosis (disease or healthy?)                  ║
║  • Email filtering (spam or not?)                           ║
║  • Customer segmentation (buy or not buy?)                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# HOW DO TREES DECIDE WHAT TO ASK?

print("\n" + "="*70)
print("HOW DECISION TREES LEARN")
print("="*70)

print("""
STEP 1: Find the BEST question to ask first
  → The question that seperate classes most clearly

STEP 2: Split data based on that question
  → Left branch: Yes answers
  → Right branch: No answers
      
STEP 3: Repeat for each branch until:
  → All samples in a leaf are same class (pure), OR
  → Reach maximum depth, OR
  → Too few samples to split further

HOW TO MEASURE "BEST" QUESTION?
      
Two Popular Methods:
      
1. GINI IMPURITY (default in scikit-learn)
   Gini = 1 - Σ(probability of each class)²
      
   Range: 0 (pure) to 0.5 (50-50 split)
      
   Example:
   • Node with 100 "Approve" + 0 "Reject" → Gini = 0 (PURE!)
   • Node with 50 "Approve" + 50 "Reject" → Gini = 0.5 (IMPURE!)
   • Node with 80 "Approve" + 20 "Reject" → Gini = 0.32 (okay)
      
2. ENTROPY (Information Gain)
   Entropy = -Σ(p × log₂(p))
      
   Range: 0 (pure) to 1 (maximum disorder)
      
   Used when you want information theory interpretation

ALGORITHM PICKS: Question that gives LOWEST impurity after split!
""")

# VISUAL DEMONSTARTION

print("\n" + "="*70)
print("VISUAL DEMONSTRATION")
print("="*70)

# Generate simple 2D classification data
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           flip_y=0.1, random_state=42)
                           
# Train a simple decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Visulaize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Decision Boundary
ax1 = axes[0]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=60,
            edgecolor='black', label='Class 0', alpha=0.7)
ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', s=60,
            edgecolor='black', label='Class 1', alpha=0.7)
ax1.set_xlabel('Feature 1', fontweight='bold', fontsize=12)
ax1.set_ylabel('Feature 2', fontweight='bold', fontsize=12)
ax1.set_title('Decision Tree: Rectangular Decision Boundaries', 
             fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Tree Structure
ax2 = axes[1]
plot_tree(tree, filled=True, feature_names=['Feature  1', 'Feature 2'],
          class_names=['Class 0', 'Class 1'], ax=ax2, fontsize=9)
ax2.set_title('Decision Tree Structure', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('01_decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_decision_tree_visualization.png")

# ADVANTAGES & DISADVANTAGES

print("\n" + "="*70)
print("DECISION TREES: PROS & CONS")
print("="*70)

print("""
✅ ADVANTAGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EASY TO UNDERSTAND & INTERPRET
   → Can draw the tree and explain to non-technical stakeholders
   → "Why was loan rejected? Because income < ₹50K AND age < 25"

2. NO DATA PREPROCESSING NEEDED
   → Works with different scales (no standardization!)
   → Handles missing values well
   → Works with categorical data directly

3. HANDLES NON-LINEAR RELATIONSHIPS
   → Can capture complex patterns
   → Unlike linear/logistic regression

4. FEATURE IMPORTANCE
   → Automatically tells you which features matter most
   → Great for feature selection

5. WORKS FOR CLASSIFICATION & REGRESSION
   → Same algorithm, different tasks


❌ DISADVANTAGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. OVERFITTING (BIGGEST PROBLEM!)
   → Without limits, tree memorizes training data
   → Poor generalization to test data
   
   Solution: Hyperparameter tuning
   • max_depth: Limit tree depth
   • min_samples_split: Minimum samples to split
   • min_samples_leaf: Minimum samples in leaf

2. UNSTABLE
   → Small changes in data → Completely different tree
   
   Solution: Use ensemble methods (Random Forest, tomorrow!)

3. BIASED TOWARD FEATURES WITH MORE CATEGORIES
   → Feature with 100 values favored over feature with 2 values
   
   Solution: Use Random Forest or balanced splitting

4. NOT GREAT FOR LINEAR RELATIONSHIPS
   → Approximates straight line with stairs
   
   When to use Linear Regression instead: If relationship is clearly linear


WHEN TO USE DECISION TREES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Use When:
- Need interpretable model (explain to business)
- Non-linear relationships
- Mixed data types (numerical + categorical)
- Feature importance matters
- Quick prototyping

❌ Avoid When:
- Need very high accuracy (use ensemble instead)
- Linear relationships (use Linear/Logistic Regression)
- High-dimensional sparse data (use Naive Bayes/SVM)
""")

# KEY HYPERPARAMETERS

print("\n" + "="*70)
print("KEY HYPERPARAMETERS (Prevent Overfitting)")
print("="*70)

hyperparams = """
MOST IMPORTANT HYPERPARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. max_depth (default: None)
   → Maximum depth of tree
   → Higher = more complex (overfitting risk)
   → Start with: 3-10 for most problems
   
2. min_samples_split (default: 2)
   → Minimum samples required to split a node
   → Higher = simpler tree
   → Start with: 20-50
   
3. min_samples_leaf (default: 1)
   → Minimum samples required in a leaf
   → Higher = smoother boundaries
   → Start with: 10-30
   
4. max_features (default: None)
   → Number of features to consider for best split
   → Lower = more randomness (good!)
   → Use: 'sqrt' for classification, 'log2' also good
   
5. criterion (default: 'gini')
   → 'gini' or 'entropy'
   → Gini is faster, entropy slightly more accurate
   → In practice: Not much difference

TUNING STRATEGY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start Simple → Add Complexity:

Step 1: max_depth=3 (simple tree)
Step 2: If underfitting, increase to 5, 7, 10
Step 3: If overfitting, add min_samples_split=20
Step 4: Add min_samples_leaf=10
Step 5: Use cross-validation to find best combo
"""

print(hyperparams)

print("\n" + "="*70)
print("SESSION 1 COMPLETE: Concepts Understood!")
print("="*70)