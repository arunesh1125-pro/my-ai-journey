"""
BATCH PREDICTION PROCESSOR
===========================
Upload CSV and get predictions for multiple customers
"""
import pandas as pd
import numpy as np

def prepare_batch_features(df):
    """Prepare uploaded data for model prediction"""

    # Required columns
    required_cols = [
        'Age', 'AnnualIncome', 'CityTier', 'TenureDays',
        'PurchaseFrequency', 'TotalSpending', 'RecencyDays',
        'SatisficationScore', 'NPSScore'
    ]

    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    

    # Calculate derived features
    df['AvgOrderValue'] = df['TotalSpending'] / (df['PurchaseFrequency'] + 1)
    df['LoyaltyPoints'] = df['PurchaseFrequency'] * 100


    # Add default values for missing optional columns
    if 'CategoriesCount' not in df.columns:
        df['CategoriesCount'] = 3
    if 'WebsiteVisits' not in df.columns:
        df['WebsiteVisits'] = 12
    if 'AppUsageHours' not in df.columns:
        df['AppUsageHours'] = 5.0
    if 'EmailOpenRate' not in df.columns:
        df['EmailOpenRate'] = 0.6
    if 'SupportTickets' not in df.columns:
        df['SupportTickets'] = 1
    if 'Gender' not in df.columns:
        df['Gender'] = 'Male'
    
    # Gender encoding
    df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
    
    # Feature order (must match training)
    feature_cols = [
        'Age', 'AnnualIncome', 'CityTier', 'TenureDays',
        'PurchaseFrequency', 'TotalSpending', 'AvgOrderValue',
        'RecencyDays', 'CategoriesCount', 'WebsiteVisits',
        'AppUsageHours', 'EmailOpenRate', 'SupportTickets',
        'SatisfactionScore', 'NPSScore', 'LoyaltyPoints',
        'Gender_Male'
    ]
    
    return df[feature_cols]

def process_batch_predictions(df, revenue_model, churn_model):
    """Generate predictions for batch upload"""
    
    X = prepare_batch_features(df)
    
    # Predictions
    revenue_pred = revenue_model.predict(X)
    churn_prob = churn_model.predict_proba(X)[:, 1]
    churn_pred = churn_model.predict(X)
    
    # Add to dataframe
    results = df.copy()
    results['Predicted_Revenue'] = revenue_pred.round(2)
    results['Churn_Probability'] = (churn_prob * 100).round(1)
    results['Churn_Risk'] = churn_pred
    
    # Risk categories
    results['Risk_Level'] = pd.cut(
        results['Churn_Probability'],
        bins=[0, 40, 70, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    # Revenue tiers
    results['Revenue_Tier'] = pd.cut(
        results['Predicted_Revenue'],
        bins=[0, 5000, 10000, np.inf],
        labels=['Standard', 'Medium', 'High Value']
    )
    
    return results
