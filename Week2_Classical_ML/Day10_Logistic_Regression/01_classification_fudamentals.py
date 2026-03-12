"""
CLASSIFICATION FUNDAMENTALS
===========================
Understanding the binary classification problem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

print("="*70)
print("CLASSIFICATION: PREDICTING CATEGORIES")
print("="*70)

# WHAT IS CLASSIFICATION

print("""
╔══════════════════════════════════════════════════════════════╗
║                 WHAT IS CLASSIFICATION?                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Task: Predict which category/class an item belongs to      ║
║                                                              ║
║  Binary Classification (2 classes):                         ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │ Class 0 (Negative):  No, False, Reject, Healthy     │   ║
║  │ Class 1 (Positive):  Yes, True, Approve, Disease    │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                              ║
║  Real-World Examples:                                        ║
║  • Email Filtering: Spam (1) vs Not Spam (0)                ║
║  • Credit Approval: Approve (1) vs Reject (0)               ║
║  • Medical Diagnosis: Disease (1) vs Healthy (0)            ║
║  • Customer Churn: Will Leave (1) vs Stay (0)               ║
║  • Fraud Detection: Fraud (1) vs Legitimate (0)             ║
║  • Loan Default: Default (1) vs Repay (0)                   ║
║                                                              ║
║  Why NOT use Linear Regression for classification?          ║
║  • Output can be any number (-∞ to +∞)                      ║
║  • We need probabilities (0 to 1)                           ║
║  • We need yes/no decisions, not continuous values          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# VISUALIZE CLASSIFICATION PROBLEM

print("\n" + "="*70)
print("VISUALIZING A CLASSIFICATION PROBLEM")
print("="*70)

# Generate sample classification data
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           flip_y=0.1, random_state=42) # n_samples=200: Creates 200 datapoints, n_feature=2: Each customer has 2 features (Feature1-Purchase Amount, Feature2-Website Visits), n_informative=2: Both features actually help predict the class, n_redudant=0: No duplicates or uncessary features, n_clusters_per_class=1: Each class forms one cluster.so, the plot will look like two main groups, flip_y=0.1: Adds 10% label noise. Will buy = 1, won't buy = 0, random_state=42

# Convert to DataFrame for easy viewing
df_demo = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df_demo['Class'] = y

print("\nExample: Customer Segmentation")
print("Features: Purchase Amount, Website Visits")
print("Classes: Will Buy (1) vs Won't Buy (0)")
print(f"\nDataset: {len(df_demo)} customers")
print(f"\nClass distribution: ")
print(df_demo['Class'].value_counts())
print(f"\nFirts 5 rows")
print(df_demo.head())

# Visulaize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Scatter plot showing two classes
class_0 = df_demo[df_demo['Class']==0]
class_1 = df_demo[df_demo['Class']==1]

axes[0].scatter(class_0['Feature_1'], class_0['Feature_2'],
                c='blue', s=80, alpha=0.6, edgecolor='black')
axes[0].scatter(class_1['Feature_1'], class_1['Feature_2'],
                c='red', s=80, alpha=0.6, edgecolor='black')
axes[0].set_xlabel('Feature 1 (Purchase Amount)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Feature 2 (Website Visits)', fontweight='bold', fontsize=12)
axes[0].set_title('Binary Classification Problem', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Show the decision boundary concept
from matplotlib.patches import Rectangle

axes[1].scatter(class_0['Feature_1'], class_0['Feature_2'],
                c='blue', s=80, alpha=0.6, edgecolors='black',
                label='Class 0', marker='o')
axes[1].scatter(class_1['Feature_1'], class_1['Feature_2'],
                c='red', s=80, alpha=0.6, edgecolor='black',
                label='Class 1', marker='^')

# Draw conceptual decision boundary
x_boundary = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # Creates 100 evenly spaced numbers b/w: minimum Feature_1 value, maximum Feature_1 value
y_boundary = 0.5 * x_boundary + 0.2 # Simplified linear boundary # It defines classifier rule - y = 0.5x + 0.2
axes[1].plot(x_boundary, y_boundary, 'g-', linewidth=3,
             label='Decision Boundary', alpha=0.8)

# Shade regions - Creates Region Coordinates: Prepares Polygon areas to fill with color
x_fill = np.array([X[:, 0].min(), X[:, 0].max(), X[:, 0].max(), X[:, 0].min()]) # This defines horizontal corners of the plot area. Eg: [-1, 3, 3, -1]
# Upper Region Coordinates
y_fill_upper = np.array([y_boundary[0], y_boundary[-1], X[:, 1].max(), X[:, 1].max()]) # This defines the area ABOVE the decision boundary. This region will be predicted as Class 1. Eg:   ________
                                                                                                                                                                                      #   /        /
                                                                                                                                                                                      #  /________/
                                                                                                                                                                                                                                                                                                                                                               
# Lower Region Coordinates
y_fill_lower = np.array([X[:, 1].min(), X[:, 1].min(), y_boundary[0], y_boundary[-1]]) # This defines the area are BELOW the boundary. This region will be predicted as Class 0. Eg:  ________
                                                                                                                                                                                  #  /       /
                                                                                                                                                                                  #  \______/

# Fill Upper Region (Class 1)
axes[1].fill(x_fill, y_fill_upper, alpha=0.2, color='red', label='Predict: Class 1') # Color the upper region are with light red. Meaning: Model prediction here -> Class 1
#Fill Lower Region (Class 0)
axes[1].fill(x_fill, y_fill_lower, alpha=0.2, color='blue', label='Predict: Class 0') # color the lower region with light blue. Meaning: Model prediction here -> Class 0


axes[1].set_xlabel('Feature 1', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Feature 2', fontweight='bold', fontsize=12)
axes[1].set_title('Classification Goal: Find Decision Boundary', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_classification_concept.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_classification_concept.png")

# CLASSIFICATION TERMINOLOGY

print("\n" + "="*70)
print("KEY CLASSIFICATION TERMINOLOGY")
print("="*70)

terminology = """
1. CLASSES / LABELS
   The categories we're predicting
   Binary: Class 0 (Negative) and Class 1 (Positive)

2. POSITIVE CLASS vs NEGATIVE CLASS
   Positive (1): The outcome we're interested in
   Negative (0): The absence of the outcome

   Examples:
   • Disease detection: Positive=Has Disease, Negative=Healthy
   • Spam filter: Positive=Spam, Negative=Not Spam
   • Loan approval: Positive=Default, Negative=Repay

3. DECISION BOUNDARY
   The line/surface that seperates the two classes
   Points on one side -> Class 0
   Points on other side -> Class 1

4. PROBABILITY
   Confidence of prediction (0 to 1)
   If P(Class=1) = 0.9 -> 90% confident it's Class 1
   If P(Class=1) = 0.3 -> 30% confident (70% it's Class 0)

5. DECISION THRESHOLD
   Probability cutoff for classification
   Default: 0.5 (if P > 0.5 -> Class 1, else Class 0)
   Can be adjusted for business needs!

6. TRUE POSITIVE (TP)
   Model correctly predicts Class 1
   Example: Predicted spam, actually spam ✓

7. TRUE NEGATIVE (TN)
   Model correctly predicts Class 0
   Example: Predicted not sapm, actually not spam ✓

8. FALSE POSITIVE (FP)
   Model predicts Class 1, but actually Class 0
   Example: Predicted spam, actually legitimate ✗
   Also called: "False Alarm"

9. FALSE NEGATIVE
   Model predicts Class 0, but actually Class 1
   Example: Predicted not spam, actually spam ✗
   Also called: "Missed Detection"
"""
print(terminology)

print("\n" + "="*70)
print("CLASSIFICATION FUNDAMENTALS COMPLETE!")
print("="*70)