"""
CAPSTONE PROJECT: Data Cleaning & Exploratory Data Analysis
============================================================
Transform messy data into ML-ready insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*70)
print("E-COMMERCE DATA ANALYSIS: CLEANING & EDA")
print("="*70)

# ============================================
# PHASE 1: LOAD DATA
# ============================================

print("\n📥 PHASE 1: LOADING DATA")
print("─"*70)

customers = pd.read_csv('../data/customers.csv')
transactions = pd.read_csv('../data/transactions_raw.csv')

# Convert dates
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

print(f"✅ Customers: {customers.shape}")
print(f"✅ Transactions: {transactions.shape}")
print()

# PHASE 2: DATA QUALITY ASSESSMENT

print("📊 PHASE 2: DATA QUALITY ASSESSMENT")
print("─"*70)

print("\nMissing Values: ")
missing_trans = transactions.isnull().sum()
missing_trans = missing_trans[missing_trans > 0]
for col, count in missing_trans.items():
    pct = count / len(transactions) * 100
    print(f"   {col:20s}: {count:>6d} ({pct:>5.2f}%)")

print("\nDuplicates: ")
dup_count = transactions.duplicated().sum()
print(f"   Duplicate rows: {dup_count} ({dup_count/len(transactions)*100:.2f}%)")

print(f"\nData Type Issues: ")
print(f"   Mixed case categories: {(transactions['category'] != transactions['category'].str.title()).sum()}")
print(f"   Whitespace in payment:  {transactions['payment_method'].str.strip().ne(transactions['payment_method']).sum()}")
print()

# PHASE 3: DATA CLEANING

print("🧹 PHASE 3: DATA CLEANING")
print("─"*70)

df = transactions.copy()

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"✅ Removed duplicates: {before} → {len(df)} rows")

# Clean text columns
df['category'] = df['category'].str.strip().str.title()
df['product'] = df['product'].str.strip().str.title()
df['payment_method'] = df['payment_method'].str.strip()
print(f"✅ Cleaned text columns")

# Handle missing values
df['rating'] = df['rating'].fillna(df['rating'].median())
df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
print(f"✅ Imputed missing values")

# Remove outliers (IQR method)
Q1 = df['total_amount'].quantile(0.25)
Q3 = df['total_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['total_amount'] < lower_bound) | (df['total_amount'] > upper_bound)]
print(f"⚠️  Found {len(outliers)} outliers in total_amount")
df = df[(df['total_amount'] >= lower_bound) & (df['total_amount'] <= upper_bound)]
print(f"✅ Removed outliers: {len(df)} rows remaining")

# Save cleaned data
df.to_csv('../data/transactions_clean.csv', index=False)
print(f"✅ Saved cleaned data")
print()

# PHASE 4: FEATURE ENGINEERING

print("⚙️  PHASE 4: FEATURE ENGINEERING")
print("─"*70)

# Time features
df['year'] = df['transaction_date'].dt.year
df['month'] = df['transaction_date'].dt.month
df['month_name'] = df['transaction_date'].dt.month_name()
df['day_of_week'] = df['transaction_date'].dt.day_name()
df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)

# Customer features (merge)
df = df.merge(customers[['customer_id', 'city', 'customer_segment', 'signup_date']],
              on='customer_id', how='left')

# Customer lifetime (days since signup at transaction time)
df['customer_lifetime_days'] = (df['transaction_date'] - df['signup_date']).dt.days

# Revenue per quantity
df['revenue_per_item'] = df['total_amount'] / df['quantity']

# Discount flag
df['has_discount'] = (df['discount_pct'] > 0).astype(int)

print("✅ Created time features")
print("✅ Merged customer data")
print("✅ Engineered business features")
print()

# PHASE 5: EXPLORATORY DATA ANALYSIS

print("🔍 PHASE 5: EXPLORATORY DATA ANALYSIS")
print("─"*70)

print("\n1. BUSINESS METRICS: ")
print("─"*40)
total_revenue = df['total_amount'].sum()
avg_order_value = df['total_amount'].mean()
total_transactions = len(df)
unique_customers = df['customer_id'].nunique()
avg_transactions_per_customer = total_transactions / unique_customers

print(f"  Total Revenue:         ₹{total_revenue:,.2f}")
print(f"  Total Transactions:    {total_transactions:,}")
print(f"  Unique Customers:      {unique_customers:,}")
print(f"  Avg Order Value:       ₹{avg_order_value:,.2f}")
print(f"  Avg Trans/Customer:    {avg_transactions_per_customer:.2f}")
print(f"  Return Rate:           {df['returned'].mean()*100:.2f}%")

print("\n2. CATEGORY PERFORMANCE: ")
print("─"*40)
category_stats = df.groupby('category').agg({
    'total_amount': ['sum', 'mean', 'count'],
    'returned': 'mean'
}).round(2)
category_stats.columns = ['Revenue', 'Avg_Order', 'Transactions', 'Return_Rate']
category_stats = category_stats.sort_values('Revenue', ascending=False)
print(category_stats)

print("\n3. TOP PRIDUCTS: ")
print("─"*40)
top_products = df.groupby('product')['total_amount'].sum().sort_values(ascending=False).head(10)
for i, (product, revenue) in enumerate(top_products.items(), 1):
    print(f"  {i:2d}. {product:15s}: ₹{revenue:>12,.2f}")

print("\n4. CUSTOMER SEGMENT ANALYSIS: ")
print("─"*40)
segment_stats = df.groupby('customer_segment').agg({
    'total_amount': ['sum', 'mean', 'count'],
    'customer_id': 'nunique'
}).round(2)
segment_stats.columns = ['Revenue', 'Avg_Order', 'Transactions', 'Customers']
print(segment_stats)

print("\n5. TIME TRENDS:")
print("─"*40)
monthly_revenue = df.groupby('month_name')['total_amount'].sum().reindex([
    'January', 'February', 'March', 'April', 'May', 'June'
])
print(monthly_revenue.round(2))

print("\n6. PAYMENT METHOD PREFERENCE: ")
print("─"*40)
payment_dist = df['payment_method'].value_counts()
for method, count in payment_dist.items():
    pct = count / len(df) * 100
    print(f"  {method:15s}: {count:>6d} ({pct:>5.2f}%)")

print("\n7. CITY-WISE PERFORMANCE: ")
print("─"*40)
city_revenue = df.groupby('city')['total_amount'].sum().sort_values(ascending=False)
for city, revenue in city_revenue.items():
    print(f"   {city:15s}: ₹{revenue:>12,.2f}")

print("\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)