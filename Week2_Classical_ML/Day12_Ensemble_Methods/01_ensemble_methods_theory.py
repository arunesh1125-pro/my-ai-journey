"""
ENSEMBLE METHODS: QUICK CONCEPTUAL UNDERSTANDING
================================================
Many models > One model (Wisdom of the Crowd)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("="*70)
print("ENSEMBLE METHODS: THE POWER OF MANY")
print("="*70)

# WHAT ARE ENSEMBLE METHODS?

print("""
╔══════════════════════════════════════════════════════════════╗
║              WHAT ARE ENSEMBLE METHODS?                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Core Idea: "Wisdom of the Crowd"                           ║
║                                                              ║
║  🌳 Single Decision Tree:                                   ║
║  One expert making all decisions                            ║
║  → Can be biased or make mistakes                           ║
║  → Overfits easily                                           ║
║                                                              ║
║  🌲🌲🌲 Ensemble (Many Trees):                               ║
║  Committee of 100 experts voting                             ║
║  → Each expert sees data differently                         ║
║  → Majority vote = final decision                            ║
║  → More robust, less overfitting                             ║
║                                                              ║
║  Real-World Analogy:                                         ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ Medical Diagnosis:                                     │ ║
║  │ • 1 doctor opinion → Might miss something             │ ║
║  │ • 5 doctors voting → More accurate diagnosis          │ ║
║  │                                                        │ ║
║  │ Same principle applies to ML models!                  │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# TWO MAIN APPROACHES

print("\n" + "="*70)
print("TWO MAIN ENSEMBLE APPROACHES")
print("="*70)

print("""
1️⃣  BAGGING (Bootstrap Aggregating)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy: Train models in PARALLEL on different random subsets

Example: RANDOM FOREST

Process:
  ┌─────────────────────────────────────────┐
  │ Original Dataset (1000 samples)         │
  └─────────────────────────────────────────┘
           │
           ├──→ Random Sample 1 → Tree 1
           ├──→ Random Sample 2 → Tree 2
           ├──→ Random Sample 3 → Tree 3
           ├──→ ...
           └──→ Random Sample 100 → Tree 100
           
  Final Prediction = Majority Vote

Key Features:
✅ Each tree sees different random subset of data
✅ Each tree uses random subset of features
✅ Reduces overfitting (main goal!)
✅ Can train trees in parallel (fast!)
✅ Great for reducing variance


2️⃣  BOOSTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy: Train models SEQUENTIALLY, each fixing previous errors

Example: GRADIENT BOOSTING, XGBoost, LightGBM

Process:
  Tree 1: Makes predictions
     ↓
  Find errors/mistakes
     ↓
  Tree 2: Focuses on fixing Tree 1's errors
     ↓
  Find remaining errors
     ↓
  Tree 3: Focuses on fixing Tree 2's errors
     ↓
  ... Continue until no improvement
     
  Final Prediction = Weighted Sum of All Trees

Key Features:
✅ Each tree learns from previous mistakes
✅ Sequential training (can't parallelize)
✅ Usually more accurate than bagging
✅ Great for reducing bias
❌ Can overfit if not tuned properly


WHEN TO USE WHICH?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Random Forest (Bagging):
→ Want interpretability + good accuracy
→ Have noisy data
→ Want to prevent overfitting
→ Need fast training
→ Good starting point

XGBoost/LightGBM (Boosting):
→ Want maximum accuracy (Kaggle competitions!)
→ Have clean data
→ Willing to tune hyperparameters carefully
→ Production systems with high performance needs
→ When Random Forest isn't accurate enough
""")

# VISUAL DEMONSTRATION

print("\n" + "="*70)
print("VISUAL COMPARISON: Single Tree vs Ensemble")
print("="*70)

# Generate data
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          flip_y=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train models
single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
gradient_boost = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

single_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
gradient_boost.fit(X_train, y_train)

# Evaluate
acc_single = accuracy_score(y_test, single_tree.predict(X_test))
acc_rf = accuracy_score(y_test, random_forest.predict(X_test))
acc_gb = accuracy_score(y_test, gradient_boost.predict(X_test))

print(f"\n📊 Accuracy Comparison:")
print(f"  Single Decision Tree:  {acc_single:.1%}")
print(f"  Random Forest:         {acc_rf:.1%}")
print(f"  Gradient Boosting:     {acc_gb:.1%}")
print(f"\n✨ Ensemble methods typically outperform single models!")

# Visualize decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [single_tree, random_forest, gradient_boost]
titles = ['Single Decision Tree', 'Random Forest (100 trees)', 
          'Gradient Boosting (100 trees)']
accuracies = [acc_single, acc_rf, acc_gb]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

for idx, (model, title, acc) in enumerate(zip(models, titles, accuracies)):
    ax = axes[idx]
    
    # Decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
              c='blue', s=60, edgecolors='black', label='Class 0', alpha=0.7)
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
              c='red', s=60, edgecolors='black', label='Class 1', alpha=0.7)
    
    ax.set_xlabel('Feature 1', fontweight='bold', fontsize=11)
    ax.set_ylabel('Feature 2', fontweight='bold', fontsize=11)
    ax.set_title(f'{title}\nAccuracy: {acc:.1%}', 
                fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_ensemble_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_ensemble_comparison.png")

# RANDOM FOREST DETAILS

print("\n" + "="*70)
print("RANDOM FOREST: KEY CONCEPTS")
print("="*70)

print("""
🌲 RANDOM FOREST = Bagging + Random Feature Selection

How It Works:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: BOOTSTRAP (Random Sampling with Replacement)
  → Create 100 random subsets of training data
  → Each subset has same size as original
  → Some samples appear multiple times, some not at all

Step 2: RANDOM FEATURES
  → For each split in each tree
  → Only consider random subset of features
  → Example: If 16 features total, consider only √16 = 4 random features
  → This adds more diversity!

Step 3: TRAIN TREES
  → Train each tree independently on its subset
  → Trees are fully grown (no pruning by default)
  → Each tree overfits its data (that's okay!)

Step 4: PREDICT
  → Classification: Each tree votes, majority wins
  → Regression: Average all tree predictions


Key Hyperparameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. n_estimators (default: 100)
   → Number of trees
   → More trees = better but slower
   → Start: 100, Increase: 200-500 if needed
   
2. max_depth (default: None)
   → Maximum depth of each tree
   → None = fully grown trees
   → Set 10-20 if overfitting
   
3. max_features (default: 'sqrt')
   → Features considered per split
   → 'sqrt' = √(total features) - good for classification
   → 'log2' = log₂(total features) - alternative
   
4. min_samples_split (default: 2)
   → Minimum samples to split node
   → Increase to 20-50 to prevent overfitting
   
5. min_samples_leaf (default: 1)
   → Minimum samples in leaf
   → Increase to 10-20 for smoother predictions

Pros:
✅ Very accurate out-of-the-box
✅ Handles missing values well
✅ Provides feature importance
✅ Reduces overfitting compared to single tree
✅ Works for classification & regression
✅ Minimal hyperparameter tuning needed

Cons:
❌ Less interpretable than single tree
❌ Slower to train and predict
❌ Larger model size (many trees)
❌ Not great for very high dimensional sparse data
""")

# ============================================
# GRADIENT BOOSTING & XGBOOST
# ============================================

print("\n" + "="*70)
print("GRADIENT BOOSTING & XGBOOST")
print("="*70)

print("""
🚀 GRADIENT BOOSTING: Sequential Error Correction

Core Idea:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instead of training trees independently:
→ Train trees one after another
→ Each new tree corrects errors of previous trees
→ Final prediction = weighted sum of all trees

Process:
  1. Start with simple prediction (e.g., mean)
  2. Calculate errors (residuals)
  3. Train tree to predict these errors
  4. Add this tree to ensemble (with small weight)
  5. Repeat until errors are minimized


⚡ XGBOOST (eXtreme Gradient Boosting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gradient Boosting on steroids! (Most popular in Kaggle)

Improvements over standard GB:
✅ Regularization (L1 & L2) → Prevents overfitting
✅ Handles missing values automatically
✅ Parallel processing → Much faster
✅ Tree pruning using max_depth → Better trees
✅ Built-in cross-validation
✅ Early stopping → Stops when no improvement

Key Hyperparameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. n_estimators (default: 100)
   → Number of boosting rounds
   → More = better but overfitting risk
   → Start: 100, Tune: 200-1000

2. learning_rate (default: 0.1)
   → How much each tree contributes
   → Lower = more trees needed but better generalization
   → Range: 0.01 - 0.3

3. max_depth (default: 3)
   → Depth of each tree
   → Shallow trees (3-6) work well for boosting
   → Deeper than Random Forest trees!

4. subsample (default: 1.0)
   → Fraction of samples for each tree
   → 0.8 = use 80% random samples
   → Adds randomness, reduces overfitting

5. colsample_bytree (default: 1.0)
   → Fraction of features per tree
   → 0.8 = use 80% random features
   → Similar to Random Forest's max_features


🌟 LightGBM (Light Gradient Boosting Machine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Even faster than XGBoost! (Microsoft's version)

Key Difference:
→ Grows trees LEAF-WISE (vs level-wise in XGBoost)
→ Can achieve same accuracy with fewer trees
→ Best for large datasets (>10,000 samples)


WHEN TO USE WHICH?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Random Forest:
→ Good baseline (start here!)
→ Less tuning needed
→ More robust to overfitting

XGBoost:
→ Need maximum accuracy
→ Kaggle competitions
→ Production systems
→ Have time to tune

LightGBM:
→ Large datasets (>100K samples)
→ Need fast training
→ Limited memory

Practical Tip:
  Try Random Forest first. If accuracy isn't enough,
  switch to XGBoost and tune hyperparameters.
""")

# ============================================
# QUICK COMPARISON TABLE
# ============================================

print("\n" + "="*70)
print("QUICK COMPARISON: All Ensemble Methods")
print("="*70)

comparison_data = {
    'Method': ['Single Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'Accuracy': ['⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
    'Speed': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐'],
    'Interpretability': ['⭐⭐⭐⭐⭐', '⭐⭐', '⭐', '⭐', '⭐'],
    'Overfitting Risk': ['High', 'Low', 'Medium', 'Low', 'Low'],
    'Tuning Needed': ['Low', 'Low', 'High', 'High', 'High'],
    'Best For': ['Simple problems', 'General use', 'Max accuracy', 'Competitions', 'Large data']
}

comparison_df = pd.DataFrame(comparison_data)
print(f"\n{comparison_df.to_string(index=False)}")

print("\n💡 Summary:")
print("  → Start with Random Forest (easiest, good results)")
print("  → Switch to XGBoost for competition-level accuracy")
print("  → Use LightGBM for very large datasets")

print("\n" + "="*70)
print("SESSION 1 COMPLETE: Ensemble Theory Understood!")
print("="*70)
