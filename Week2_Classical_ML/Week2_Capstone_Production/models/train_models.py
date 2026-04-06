"""
ML MODEL TRAINING PIPELINE
===========================
Train all models: Regression, Classification, Clustering
"""

import pandas as pd
import numpy as np
import joblib # It has set of tools to optimize the large datasets of numpy arrays

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "retail_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, r2_score,
                            accuracy_score, precision_score,
                            recall_score, f1_score)

print("="*70)
print("TRAINING ML MODELS")
print("="*70)

# LOAD DATA

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded {len(df)} records")

# PREPARE FEATURES

feature_cols = [
    'Age', 'AnnualIncome', 'CityTier', 'TenureDays',
    'PurchaseFrequency', 'TotalSpending', 'AvgOrderValue',
    'RecencyDays', 'CategoriesCount', 'WebsiteVisits',
    'AppUsageHours', 'EmailOpenRate', 'SupportTickets',
    'SatisfactionScore', 'NPSScore', 'LoyaltyPoints'
]

# Generate encoding
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
feature_cols.append('Gender_Male')

X = df[feature_cols].copy()

# MODEL 1: REVENUE PREDICTION (Regression)

print("\n" + "="*70)
print("MODEL 1: REVENUE PREDICTION (Random Forest Regressor)")
print("="*70)

y_revenue = df['NextMonthRevenue']

X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(
    X, y_revenue, test_size=0.2, random_state=42
)

# Train
rf_revenue = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    random_state=42,
    n_jobs=1 # -1 means uses all proeccsors to process
)
rf_revenue.fit(X_train_rev, y_train_rev)

# Evaluate
y_pred_rev = rf_revenue.predict(X_test_rev)
mae = mean_absolute_error(y_test_rev, y_pred_rev)
r2 = r2_score(y_test_rev, y_pred_rev)

print(f"\n✅ Revenue Model Trained!")
print(f"  MAE: ₹{mae:,.2f}")
print(f"  R² Score: {r2:.3f}")

# Save
joblib.dump(rf_revenue, 'revenue_model.pkl')
print(f"  Saved: revenue_model.pkl")

# MODEL 2: CHURN PREDICTION (Classification)

print("\n" + "="*70)
print("MODEL 2: CHURN PREDICTION (Random Forest Classifier)")
print("="*70)

y_churn = df['WillChurn']

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

# Train
rf_churn = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)
rf_churn.fit(X_train_churn, y_train_churn)

# Evaluate
y_proba_churn = rf_churn.predict_proba(X_test_churn)[:, 1]
y_pred_churn = (y_proba_churn >= 0.30).astype(int)

accuracy = accuracy_score(y_test_churn, y_pred_churn)
precision = precision_score(y_test_churn, y_pred_churn)
recall = recall_score(y_test_churn, y_pred_churn)
f1 = f1_score(y_test_churn, y_pred_churn)

print(f"\n✅ Churn Model Trained!")
print(f"  Accuracy: {accuracy:.1%}")
print(f"  Precision: {precision:.1%}")
print(f"  Recall: {recall:.1%}")
print(f"  F1-Score: {f1:.3f}")

# Save
joblib.dump(rf_churn, 'churn_model.pkl')
print(f"  Saved: churn_model.pkl")

# MODEL 3: CUSTOMER SEGMENTATION (Clustering)

print("\n" + "="*70)
print("MODEL 3: CUSTOMER SEGMENTATION (K-Means)")
print("="*70)

# Scale features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add to dataframe
df['Cluster'] = clusters

print(f"\n✅ Segmentation Model Trained!")
print(f"  Clusters: 4")
print(f"\n  Cluster Distribution:")
for i in range(4):
    count = (clusters == i).sum()
    print(f"   Cluster {i}: {count:,} customers ({count/len(df)*100:.1f}%)")

# Save
import os
os.makedirs("models", exist_ok=True)

joblib.dump(kmeans, MODEL_DIR / 'segment_model.pkl')
joblib.dump(rf_revenue, MODEL_DIR / 'revenue_model.pkl')
joblib.dump(rf_churn, MODEL_DIR / 'churn_model.pkl')
joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

print("  Saved: segment_model.pkl")
print("  Saved: scaler.pkl")

# SAVE FEATURE IMPORTANCE

feature_importance_revenue = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_revenue.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance_churn = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_churn.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance_revenue.to_csv(MODEL_DIR / 'feature_importance_revenue.csv', index=False)
feature_importance_churn.to_csv(MODEL_DIR / 'feature_importance_churn.csv', index=False)

print("\n✅ Feature importance saved")

# SAVE UPDATED DATA (with clusters)

df.to_csv(MODEL_DIR / 'retail_data_with_predictions.csv', index=False)
print(f"\n✅ Updated data saved with cluster labels")

print("\n" + "="*70)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*70)
print("\n📁 Files created:")
print("  • revenue_model.pkl")
print("  • churn_model.pkl")
print("  • segment_model.pkl")
print("  • scaler.pkl")
print("  • feature_importance_revenue.csv")
print("  • feature_importance_churn.csv")
print("  • retail_data_with_predictions.csv")