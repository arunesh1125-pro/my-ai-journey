"""
UNSUPERVISED LEARNING: CLUSTERING
==================================
Find hidden patterns WITHOUT labels!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score # Calculate the goodness of clustering
from scipy.cluster.hierarchy import dendrogram, linkage

print("="*70)
print("UNSUPERVISED LEARNING: CLUSTERING")
print("="*70)

# WHAT IS UNSUPERVISED LEARNING?

print("""
╔══════════════════════════════════════════════════════════════╗
║         SUPERVISED vs UNSUPERVISED LEARNING                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  SUPERVISED LEARNING (Days 8-12):                           ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ We have LABELS (known answers)                         │ ║
║  │                                                        │ ║
║  │ Example:                                               │ ║
║  │ • Customer data + "Did they churn?" (Yes/No)          │ ║
║  │ • Email + "Is it spam?" (Yes/No)                      │ ║
║  │ • House features + "Price?" (₹50 lakh)                │ ║
║  │                                                        │ ║
║  │ Goal: Learn from labeled examples                      │ ║
║  │       Predict labels for new data                      │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  UNSUPERVISED LEARNING (Today):                             ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ NO LABELS (no known answers)                           │ ║
║  │                                                        │ ║
║  │ Example:                                               │ ║
║  │ • Customer data (age, income, spending...)            │ ║
║  │ • NO label saying "VIP" or "Budget" customer          │ ║
║  │                                                        │ ║
║  │ Goal: Find HIDDEN PATTERNS                             │ ║
║  │       Group similar items together                     │ ║
║  │       Discover natural segments                        │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  Why Unsupervised?                                           ║
║  • Often we DON'T have labels                               ║
║  • Labels are expensive to create                           ║
║  • Want to discover unknown patterns                        ║
║  • Exploratory data analysis                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# WHAT IS CLUSTERING

print("\n" + "="*70)
print("WHAT IS CLUSTERING?")
print("="*70)

print("""
CLUSTERING: Group similar items together

Real-World Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CUSTOMER SEGMENTATION (Marketing)
   → Group customers by behavior
   → No one told us "this is a VIP customer"
   → ML finds: "These 1,000 customers behave similarly"
   
   Discovered Segments:
   • High spenders, frequent buyers → VIP
   • Low spenders, frequent buyers → Bargain hunters
   • High spenders, rare buyers → Luxury shoppers
   • Low spenders, rare buyers → Window shoppers

2. DOCUMENT ORGANIZATION
   → Group similar news articles
   → Cluster: Politics, Sports, Tech, Entertainment
   → No one labeled each article

3. IMAGE SEGMENTATION
   → Group similar pixels
   → Separate foreground from background
   → Medical imaging: Detect tumors

4. ANOMALY DETECTION
   → Find outliers (don't fit any cluster)
   → Fraud detection: Unusual transactions
   → Network intrusion: Abnormal traffic

5. RECOMMENDATION SYSTEMS
   → "Customers who bought X also bought Y"
   → Find customer clusters with similar tastes


HOW CLUSTERING WORKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Points with NO labels
  [Customer 1: Age=25, Income=30K, Spending=5K]
  [Customer 2: Age=24, Income=32K, Spending=4.5K]
  [Customer 3: Age=55, Income=80K, Spending=15K]
  [Customer 4: Age=54, Income=85K, Spending=16K]

Output: Groups (Clusters)
  Cluster 0: [Customer 1, Customer 2] → "Young Budget"
  Cluster 1: [Customer 3, Customer 4] → "Mature High-Value"

Key Idea: 
  "Similar" = Close together in feature space
  Distance metrics: Euclidean, Manhattan, Cosine
""")

# K-MEANS CLUSTERING

print("\n" + "="*70)
print("K-MEANS CLUSTERING (Most Popular!)")
print("="*70)

print("""
K-MEANS: Partition data into K clusters

How It Works:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: CHOOSE K (number of clusters)
  Example: K=3 (want 3 customer segments)

Step 2: INITIALIZE K centroids randomly
  Place 3 random points as "cluster centers"

Step 3: ASSIGN points to nearest centroid
  For each customer:
    → Calculate distance to all 3 centroids
    → Assign to closest one

Step 4: UPDATE centroids
  For each cluster:
    → New centroid = average of all points in cluster

Step 5: REPEAT Steps 3-4
  Until centroids stop moving (convergence)


Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Iteration 0: Random centroids ⭐⭐⭐
Iteration 1: Points assigned, centroids move
Iteration 2: Points reassigned, centroids move
Iteration 3: Points reassigned, centroids move
...
Iteration 10: CONVERGED! Centroids stable ✅


Key Hyperparameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. n_clusters (K)
   → Number of clusters
   → Most important decision!
   → Use Elbow Method to find optimal K

2. init (default: 'k-means++')
   → How to initialize centroids
   → 'k-means++' = smarter initialization (better results)

3. max_iter (default: 300)
   → Maximum iterations
   → Usually converges in <100

4. n_init (default: 10)
   → Run algorithm 10 times with different initializations
   → Pick best result (lowest inertia)


Pros:
✅ Fast and scalable
✅ Works well with large datasets
✅ Easy to understand and implement
✅ Guaranteed to converge

Cons:
❌ Need to choose K beforehand
❌ Sensitive to outliers
❌ Assumes spherical clusters (same size/shape)
❌ Random initialization can give different results
""")

# ELBOW METHOD

print("\n" + "="*70)
print("ELBOW METHOD: Finding Optimal K")
print("="*70)

print("""
Problem: How many clusters should we use?

Elbow Method:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Try different values of K (e.g., K=2, 3, 4, 5, 6, 7, 8)

2. For each K, calculate INERTIA (Within-Cluster Sum of Squares)
   Inertia = Sum of squared distances from points to their centroid
   → Lower inertia = tighter clusters = better

3. Plot K vs Inertia

4. Look for "ELBOW" - point where improvement slows down

Example:
   Inertia
     |
1000 |●
     |
 800 |  ●
     |
 600 |    ●
     |      ●
 400 |        ●___●___●___●
     |
   0 +─────────────────────── K
     2  3  4  5  6  7  8  9
     
     ↑ ELBOW at K=5!
     
Interpretation:
- K=2: Too few clusters (high inertia)
- K=5: Sweet spot (elbow) ← CHOOSE THIS
- K=9: Too many clusters (marginal improvement)

Rule: Pick K at the "elbow" where adding more clusters
      gives diminishing returns
""")

# VISUAL DEMONSTRATION

print("\n" + "="*70)
print("VISUAL DEMONSTRATION")
print("="*70)

# Generate sample data with 3 natural clusters
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                       cluster_std=0.8, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different K Values
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

print(f"\nElbow Method Results:")
print(f"{'K':<5} {'Inertia':<15} {'Silhouette Score':<20}")
print("-" * 40)
for k, inertia, sil in zip(K_range, inertias, silhouette_scores):
    print(f"{k:<5} {inertia:<15.2f} {sil:<20.3f}")

# Fit K-Means with optimal K=3
kmeans_optimal = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans_optimal.fit_predict(X_scaled)
centroids = kmeans_optimal.cluster_centers_

# Visualizations
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Original Data (No Labels)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], s=60, alpha=0.6, 
           edgecolors='black', c='gray')
ax1.set_xlabel('Feature 1 (Standardized)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Feature 2 (Standardized)', fontweight='bold', fontsize=11)
ax1.set_title('Original Data (No Labels)', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.text(0.5, 0.95, 'UNSUPERVISED: We don\'t know groups!', 
        transform=ax1.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: K-Means Result
ax2 = fig.add_subplot(gs[0, 1])
colors = ['#e74c3c', '#3498db', '#2ecc71']
for i in range(3):
    cluster_points = X_scaled[clusters == i]
    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               s=60, alpha=0.6, edgecolors='black', c=colors[i],
               label=f'Cluster {i}')
ax2.scatter(centroids[:, 0], centroids[:, 1], 
           s=300, c='black', marker='X', edgecolors='yellow',
           linewidths=3, label='Centroids')
ax2.set_xlabel('Feature 1 (Standardized)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Feature 2 (Standardized)', fontweight='bold', fontsize=11)
ax2.set_title('K-Means Clustering (K=3)', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.95, 'ML Found 3 Groups Automatically!', 
        transform=ax2.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Elbow Method
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(K_range, inertias, 'o-', linewidth=3, markersize=10, 
        color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
ax3.axvline(x=3, color='green', linestyle='--', linewidth=2, 
           alpha=0.7, label='Optimal K=3')
ax3.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold', fontsize=11)
ax3.set_title('Elbow Method', fontweight='bold', fontsize=13)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.annotate('ELBOW', xy=(3, inertias[1]), xytext=(5, inertias[1] + 20),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, fontweight='bold', color='green')

# Plot 4: Silhouette Score
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(K_range, silhouette_scores, 's-', linewidth=3, markersize=10,
        color='#3498db', markerfacecolor='white', markeredgewidth=2)
ax4.axvline(x=3, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Silhouette Score', fontweight='bold', fontsize=11)
ax4.set_title('Silhouette Analysis', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.text(0.5, 0.05, 'Higher = Better separated clusters', 
        transform=ax4.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Plot 5: Cluster sizes
ax5 = fig.add_subplot(gs[1, 1])
cluster_sizes = pd.Series(clusters).value_counts().sort_index()
bars = ax5.bar(range(3), cluster_sizes.values, color=colors,
              edgecolor='black', linewidth=2)
ax5.set_xlabel('Cluster', fontweight='bold', fontsize=11)
ax5.set_ylabel('Number of Points', fontweight='bold', fontsize=11)
ax5.set_title('Cluster Distribution', fontweight='bold', fontsize=13)
ax5.set_xticks(range(3))
ax5.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', 
            fontweight='bold', fontsize=11)

# Plot 6: Info box
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

info_text = f"""
╔════════════════════════════════════╗
║       K-MEANS SUMMARY              ║
╠════════════════════════════════════╣
║                                    ║
║ Algorithm: K-Means Clustering      ║
║ Optimal K: 3 clusters              ║
║                                    ║
║ Metrics:                           ║
║ • Inertia: {inertias[1]:.2f}                 ║
║ • Silhouette: {silhouette_scores[1]:.3f}               ║
║                                    ║
║ Cluster Sizes:                     ║
║ • Cluster 0: {cluster_sizes[0]} points          ║
║ • Cluster 1: {cluster_sizes[1]} points          ║
║ • Cluster 2: {cluster_sizes[2]} points          ║
║                                    ║
║ Method Used:                       ║
║ • Elbow Method ✓                   ║
║ • Silhouette Analysis ✓            ║
║                                    ║
║ Result:                            ║
║ Found 3 natural groups             ║
║ without any labels!                ║
║                                    ║
╚════════════════════════════════════╝
"""

ax6.text(0.5, 0.5, info_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('K-MEANS CLUSTERING: COMPLETE DEMONSTRATION',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('01_kmeans_demonstration.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_kmeans_demonstration.png")

# HIERARCHICAL CLUSTERING

print("\n" + "="*70)
print("HIERARCHICAL CLUSTERING")
print("="*70)

print("""
HIERARCHICAL CLUSTERING: Build a tree of clusters

How It Works (Agglomerative - Bottom-Up):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Start with each point as its own cluster
  → 300 points = 300 clusters

Step 2: Merge the two CLOSEST clusters
  → 299 clusters

Step 3: Repeat until one big cluster
  → Keep merging closest pairs

Result: DENDROGRAM (tree diagram)
  → Shows how clusters were merged
  → Can cut tree at any height to get K clusters


Linkage Methods (How to measure distance between clusters):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SINGLE (minimum distance)
   → Distance = closest two points
   → Can create long chains

2. COMPLETE (maximum distance)
   → Distance = farthest two points
   → Creates compact clusters

3. AVERAGE
   → Distance = average of all pairs
   → Balanced approach

4. WARD (default) ⭐
   → Minimize variance when merging
   → Best for most use cases


Pros vs K-Means:
✅ Don't need to specify K beforehand
✅ Creates hierarchy (dendrogram)
✅ Can handle non-spherical clusters
✅ Deterministic (same result every time)

Cons vs K-Means:
❌ Slower (O(n² log n) vs O(n))
❌ Not scalable to large datasets
❌ Can't undo merges (greedy algorithm)
""")

# Quick hierarchical clustering demo
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_clusters = hierarchical.fit_predict(X_scaled)

print(f"\nHierarchical Clustering Result:")
print(f"  Silhouette Score: {silhouette_score(X_scaled, hier_clusters):.3f}")
print(f"  Comparison with K-Means: {silhouette_score(X_scaled, clusters):.3f}")

# COMPARISON SUMMARY

print("\n" + "="*70)
print("ALGORITHM COMPARISON")
print("="*70)

comparison = """
┌─────────────────┬──────────────┬────────────────────┐
│ Aspect          │ K-Means      │ Hierarchical       │
├─────────────────┼──────────────┼────────────────────┤
│ Speed           │ Fast ⭐⭐⭐⭐⭐    │ Slow ⭐⭐           │
│ Scalability     │ Large data ✓ │ Small data only    │
│ Need K?         │ Yes          │ No (from dendrogram)│
│ Cluster Shape   │ Spherical    │ Any shape          │
│ Deterministic   │ No           │ Yes                │
│ Best Use        │ General      │ Small data, explore│
└─────────────────┴──────────────┴────────────────────┘

Recommendation:
→ Start with K-Means (faster, works well for most cases)
→ Use Hierarchical for small datasets when you want to explore
  different numbers of clusters via dendrogram
"""

print(comparison)

print("\n" + "="*70)
print("SESSION 1 COMPLETE: Clustering Theory Understood!")
print("="*70)