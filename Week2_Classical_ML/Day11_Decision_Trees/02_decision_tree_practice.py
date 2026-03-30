"""
DECISION TREES: HANDS-ON PRACTICE
==================================
Quick practical implementation with scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

print("="*70)
print("DECISION TREES: HANDS-ON PRACTICE")
print("="*70)

# EXAMPLE 1: CLASSIFICATION (Iris Dataset)

print("\n" + "="*70)
print("EXAMPLE 1: CLASSIFICATION - Iris Species")
print("="*70)

# Load famous Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Classes: {iris.target_names}")
print(f"Features: {iris.feature_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL 1: DEFAULT DECISION TREE (Overfit!)

print("\n📊 MODEL 1: Default Tree (No Limits)")
print("-" * 50)

tree_default = DecisionTreeClassifier(random_state=42)
tree_default.fit(X_train, y_train)

train_acc = tree_default.score(X_train, y_train)
test_acc = tree_default.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.1%}")
print(f"Test Accuracy:  {test_acc:.1%}")
print(f"Tree Depth: {tree_default.get_depth()}")
print(f"Number of Leaves: {tree_default.get_n_leaves()}")

if train_acc - test_acc > 0.05:
    print("⚠️  OVERFITTING DETECTED!")
    print("   Train much better than test = memorizing data")
print("-" * 50)

# METHOD 2: CONTROLLED TREE (Better!)

print("\n📊 MODEL 2: Controlled Tree (Hyperparameters)")
print("-" * 50)

tree_tuned = DecisionTreeClassifier(
    max_depth=3,                # Limited depth
    min_samples_split=20,       # Need 20 samples to split
    min_samples_leaf=10,        # Need 10 samples in leaf
    random_state=42
)
tree_tuned.fit(X_train, y_train)

train_acc_tuned = tree_tuned.score(X_train, y_train)
test_acc_tuned = tree_tuned.score(X_test, y_test)

print(f"Train Accuracy: {train_acc_tuned:.1%}")
print(f"Test Accuracy:  {test_acc_tuned:.1%}")
print(f"Tree Depth: {tree_tuned.get_depth()}")
print(f"Number of Leaves: {tree_tuned.get_n_leaves()}")


if abs(train_acc_tuned - test_acc_tuned) < 0.05:
    print("✅ GOOD GENERALIZATION!")
    print("   Train and test similar = not overfitting")

# FEATURE IMPORTANCE

print("\n📊 FEATURE IMPORTANCE")
print("-" * 50)

importance_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance':tree_tuned.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))
print("\n💡 Higher importance = more useful for classification")

# VISUALIZATIONS

fig = plt.figure(figsize=(16, 10))

# Plot 1: Default Tree (Complex)
ax1 = plt.subplot(2, 2, 1)
plot_tree(tree_default, filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          ax=ax1, fontsize=7)
ax1.set_title(f'Default Tree (Depth={tree_default.get_depth()}, Overfit Risk)', 
             fontweight='bold', fontsize=12)

# Plot 2: Tuned Tree (Simple)
ax2 = plt.subplot(2, 2, 2)
plot_tree(tree_tuned, filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          ax=ax2, fontsize=9)
ax2.set_title(f'Tuned Tree (Depth={tree_tuned.get_depth()}, Better Generalization)',
             fontweight='bold', fontsize=12)

# Plot 3: Model Comparison
ax3 = plt.subplot(2, 2, 3)
models = ['Default Tree', 'Tuned Tree']
train_scores = [train_acc, train_acc_tuned]
test_scores = [test_acc, test_acc_tuned]

x = np.arange(len(models))
width = 0.35

bars1 = ax3.bar(x - width/2, train_scores, width, label='Train', 
               color='#3498db', edgecolor='black', linewidth=2)
bars2 = ax3.bar(x + width/2, test_scores, width, label='Test',
               color='#2ecc71', edgecolor='black', linewidth=2)

ax3.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax3.set_title('Train vs Test Accuracy', fontweight='bold', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend(fontsize=10)
ax3.set_ylim(0.8, 1.0)
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=10)

# Plot 4: Feature Importance
ax4 = plt.subplot(2, 2, 4)
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
bars = ax4.barh(importance_df['Feature'], importance_df['Importance'],
               color=colors, edgecolor='black', linewidth=2)
ax4.set_xlabel('Importance', fontweight='bold', fontsize=11)
ax4.set_title('Feature Importance', fontweight='bold', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

for i, (feature, imp) in enumerate(zip(importance_df['Feature'],
                                       importance_df['Importance'])):
    ax4.text(imp + 0.01, i, f'{imp:.3f}', va='center',
             fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('02_decision_tree_practice.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 02_decision_tree_practice.png")

# CROSS-VALIDATION

print("\n📊 CROSS-VALIDATION (5-Fold)")
print("-" * 50)

cv_scores = cross_val_score(tree_tuned, X, y, cv=5)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print("💡 Consistent across folds = reliable model")

# QUICK COMPARISON: TREE VS LOGISTIC REGRESSION

print("\n📊 QUICK COMPARISON: Decision Tree vs Logistic Regression")
print("-" * 50)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

lr_test_acc = log_reg.score(X_test, y_test)

print(f"Decision Tree Test Accuracy: {test_acc_tuned:.1%}")
print(f"Logistic Regression Test Accuracy: {lr_test_acc:.1%}")

print("\n💡 When to use which?")
print("  Decision Tree: Non-linear patterns, interpretability needed")
print("  Logistic Reg: Linear separable data, need probabilities")

print("\n" + "="*70)
print("SESSION 2 COMPLETE: Hands-on Practice Done!")
print("="*70)
