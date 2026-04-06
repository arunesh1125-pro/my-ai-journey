"""
COMPREHENSIVE RETAIL DATASET GENERATOR
=======================================
Generates realistic retail business data for ML training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "retail_data.csv"

print("Generating Comprehensive Retail Dataset...")

np.random.seed(42)
n_customers = 3000

# CUSTOMER DEOGRAPHICS

customer_ids = [f'CUST{i:05d}' for i in range(1, n_customers + 1)]
age = np.random.normal(38, 12, n_customers).clip(18, 75)
gender = np.random.choice(['Male', 'Female'], n_customers, p=[0.48, 0.52])
income = np.random.lognormal(11.2, 0.6, n_customers).clip(25000, 500000)
city_tier = np.random.choice([1, 2, 3], n_customers, p=[0.25, 0.45, 0.30])

# PURCHASE BEHAVIOUR

# Tenure (days since first purchase)
tenure_days = np.random.exponential(500, n_customers).clip(30, 2000)

# Purchase frequency
# Better purchase frequency spread
purchase_frequency = np.random.negative_binomial(6, 0.4, n_customers).clip(1, 40)

# Total spending (influenced by income, tenure, frequency)
base_spending = (
    income * 0.22 +  # 22% of income
    tenure_days * 8 +  # More tenure = more spending
    purchase_frequency * 1800 +  # More purchases = more spending
    np.random.normal(0, 8000, n_customers)
).clip(5000, 250000)

# Average order value
avg_order_value = base_spending / purchase_frequency

# Last purchase recency (days)
recency_days = np.random.exponential(35, n_customers).clip(0, 365)

# Product categories purchased
categories_count = np.random.poisson(4, n_customers).clip(1, 10)

# ENGAGEMENT METRICS

website_visits = np.random.poisson(12, n_customers).clip(1, 100)
app_usage_hours = np.random.exponential(5, n_customers).clip(0, 50)
email_open_rate = np.random.beta(3, 2, n_customers).clip(0.1, 0.95)
support_tickets = np.random.poisson(1.5, n_customers).clip(0, 15)

# SATISFACTION & LOYALTY

satisfaction_score = np.random.normal(7.5, 1.5, n_customers).clip(1, 10)
nps_score = np.random.choice(range(0, 11), n_customers)
loyalty_points = (purchase_frequency * 100 + base_spending * 0.01).clip(0, 10000)

# GENERATE TARGET VARIABLES

# 1. NEXT MONTH REVENUE (Regression)
next_month_revenue = (
    0.12 * avg_order_value * purchase_frequency +
    0.015 * base_spending +
    -1.5 * recency_days +
    120 * satisfaction_score +
    500 * (city_tier == 1) +
    np.random.normal(0, 2000, n_customers)
).clip(2000, 60000)

# 2. CHURN PROBABILITY (Classification)
churn_logit = (
    -1.25 +   # lower baseline churn
    -0.003 * (age - 35) +
    -0.0000015 * income +
    -0.012 * purchase_frequency +
    0.010 * recency_days +   # reduced further
    -0.18 * satisfaction_score +
    -0.045 * nps_score +
    0.16 * support_tickets +
    0.65 * (tenure_days < 180)
)
churn_probability = 1 / (1 + np.exp(-churn_logit))
will_churn = np.random.binomial(1, churn_probability)
# will_churn = (churn_probability > np.random.uniform(0, 1, n_customers)).astype(int)

# 3. CUSTOMER SEGMENTS (for clustering visualization)
# We'll generate 4 natural segments
segment_labels = []
for i in range(n_customers):
    if base_spending[i] > 75000 and purchase_frequency[i] > 10:
        segment_labels.append('VIP')
    elif income[i] > 130000 and purchase_frequency[i]>= 5:
        segment_labels.append('High Potential')
    elif purchase_frequency[i] > 8 and satisfaction_score[i] > 7:
        segment_labels.append('Loyal')
    else:
        segment_labels.append('At Risk')

# CREATE DATAFRAME

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Age': age.round(0).astype(int),
    'Gender': gender,
    'AnnualIncome': income.round(0).astype(int),
    'CityTier': city_tier,
    'TenureDays': tenure_days.round(0).astype(int),
    'PurchaseFrequency': purchase_frequency,
    'TotalSpending': base_spending.round(2),
    'AvgOrderValue': avg_order_value.round(2),
    'RecencyDays': recency_days.round(0).astype(int),
    'CategoriesCount': categories_count,
    'WebsiteVisits': website_visits,
    'AppUsageHours': app_usage_hours.round(1),
    'EmailOpenRate': email_open_rate.round(3),
    'SupportTickets': support_tickets,
    'SatisfactionScore': satisfaction_score.round(1),
    'NPSScore': nps_score,
    'LoyaltyPoints': loyalty_points.round(0).astype(int),
    'NextMonthRevenue': next_month_revenue.round(2),
    'WillChurn': will_churn,
    'TrueSegment': segment_labels
})

# SAVE DATA

df.to_csv(DATA_PATH, index=False)
print(f"✅ Generated {len(df)} customer records")
print(f"✅ Saved to: retail_data.csv")

# Print summary
print(f"\n📊 Dataset Summary:")
print(f"  Customers: {len(df):,}")
print(f"  Features: {df.shape[1] - 3}")  # Exclude targets
print(f"  Churn Rate: {df['WillChurn'].mean():.1%}")
print(f"  Avg Revenue: ₹{df['NextMonthRevenue'].mean():,.0f}")
print(f"\nSegment Distribution:")
print(df['TrueSegment'].value_counts())