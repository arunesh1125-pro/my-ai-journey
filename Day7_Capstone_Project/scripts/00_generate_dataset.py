"""
CAPSTONE PROJECT: E-Commerce Customer Behavior Analysis
========================================================
Step 1: Generate realistic e-commerce dataset

Dataset simulates 6 months of customer transactions for an online store
selling electronics, clothing, and home goods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("GENERATING E-COMMERCE DATASET")
print("="*70)

np.random.seed(42)

# CONFIGURATION

n_customers = 2000
n_transactions = 15000
start_date = datetime(2026, 1, 1)
end_date = datetime(2026, 6, 30)

print(f"\nDataset Configuration: ")
print(f"    Customers: {n_customers:,}")
print(f"    Transactions: {n_transactions:,}")
print(f"    Period: {start_date.date()} to {end_date.date()}")
print()

# GENERATE CUSTOMERS

print("Generating customer profiles...")

customer_ids = [f'CUST{i:05d}' for i in range(1, n_customers + 1)]
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
          'Pune', 'Ahmedabad']
customer_segments = ['Premium', 'Regular', 'Occassional']
device_types = ['Mobile', 'Desktop', 'Tablet']

customers = pd.DataFrame({
    'customer_id': customer_ids,
    'city': np.random.choice(cities, n_customers),
    'signup_date': [start_date + timedelta(days=np.random.randint(-365, 0))
                                           for _ in range(n_customers)],
    'customer_segment': np.random.choice(customer_segments, n_customers,
                                         p=[0.15, 0.60, 0.25]),
    'preferred_device': np.random.choice(device_types, n_customers, 
                                          p=[0.60, 0.30, 0.10])                                        
})

print(f"✅ Generated {len(customers):,} customer profiles")

# GENERATE TRANSACTIONS

print("Generating transactions...")

# Product Catalog
product_categories = {
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Tablet'],
    'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
    'Home': ['Furniture', 'Kitchenware', 'Decor', 'Bedding', 'Appliance']
}

# Price range by category
price_ranges = {
    'Electronics': (5000, 80000),
    'Clothing': (500, 5000),
    'Home': (1000, 30000)
}

transactions = []

for _ in range(n_transactions):
    # Select random customer
    customer_id = np.random.choice(customer_ids)
    customer_row = customers[customers['customer_id'] == customer_id].iloc[0]
    
    # Random date within range
    random_days = np.random.randint(0, (end_date - start_date).days)
    transaction_date = start_date + timedelta(days=random_days)
    
    # Ensure transaction after signup
    if transaction_date < customer_row['signup_date']:
        transaction_date = customer_row['signup_date'] + timedelta(days=np.random.randint(1, 30))
    
    # Category (segment affects category preference)
    if customer_row['customer_segment'] == 'Premium':
        category = np.random.choice(['Electronics', 'Clothing', 'Home'], p=[0.5, 0.3, 0.2])
    else:
        category = np.random.choice(['Electronics', 'Clothing', 'Home'], p=[0.3, 0.4, 0.3])
    
    product = np.random.choice(product_categories[category])
    
    # Price (segment affects price)
    base_price = np.random.uniform(*price_ranges[category])
    if customer_row['customer_segment'] == 'Premium':
        price = base_price * np.random.uniform(1.2, 1.5)
    elif customer_row['customer_segment'] == 'Occasional':
        price = base_price * np.random.uniform(0.7, 0.9)
    else:
        price = base_price
    
    # Quantity (usually 1, sometimes more)
    if category == 'Clothing':
        quantity = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    else:
        quantity = np.random.choice([1, 2], p=[0.85, 0.15])
    
    # Payment method (segment affects method)
    if customer_row['customer_segment'] == 'Premium':
        payment_method = np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Wallet'],
                                          p=[0.5, 0.2, 0.2, 0.1])
    else:
        payment_method = np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Wallet'],
                                          p=[0.2, 0.3, 0.3, 0.2])
    
    # Discount (20% of transactions have discount)
    discount_pct = np.random.choice([0, 5, 10, 15, 20], p=[0.80, 0.10, 0.05, 0.03, 0.02])
    
    # Delivery time (days)
    delivery_days = np.random.choice([1, 2, 3, 4, 5, 7], p=[0.1, 0.3, 0.3, 0.2, 0.08, 0.02])
    
    # Rating (1-5 stars, skewed toward positive)
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.03, 0.10, 0.35, 0.50])
    
    # Return status (5% return rate)
    returned = np.random.choice([0, 1], p=[0.95, 0.05])
    
    transactions.append({
        'transaction_id': f'TXN{len(transactions)+1:06d}',
        'customer_id': customer_id,
        'transaction_date': transaction_date,
        'category': category,
        'product': product,
        'quantity': quantity,
        'unit_price': round(price, 2),
        'discount_pct': discount_pct,
        'payment_method': payment_method,
        'delivery_days': delivery_days,
        'rating': rating if not returned else np.nan,
        'returned': returned,
        'device': customer_row['preferred_device']
    })

df_transactions = pd.DataFrame(transactions)

# Calculate total amount
df_transactions['discount_amount'] = (
    df_transactions['unit_price'] * 
    df_transactions['quantity'] * 
    df_transactions['discount_pct'] / 100
).round(2)

df_transactions['total_amount'] = (
    df_transactions['unit_price'] * df_transactions['quantity'] - 
    df_transactions['discount_amount']
).round(2)

print(f"✅ Generated {len(df_transactions):,} transactions")

# ADD DATA QUALITY ISSUES (Realistic!)

print("\nAdding realistic data quality issues...")

# Missing values
df_transactions.loc[df_transactions.sample(frac=0.03).index, 'rating'] = np.nan  # .sample() used to pick random pct of rows and set of specific columns to np.nan
df_transactions.loc[df_transactions.sample(frac=0.01).index, 'delivery_days'] = np.nan

# Duplicates (1% accidental duplicates)
dup_indices = df_transactions.sample(frac=0.01).index
df_transactions = pd.concat([df_transactions, df_transactions.loc[dup_indices]])
df_transactions = df_transactions.reset_index(drop=True)

# Outliers (a few extreme values)
outlier_indices = df_transactions.sample(n=20).index
df_transactions.loc[outlier_indices, 'total_amount'] *= np.random.uniform(5, 10, 20)

# Mixed case in categories
sample_idx = df_transactions.sample(frac=0.02).index
df_transactions.loc[sample_idx, 'category'] = \
    df_transactions.loc[sample_idx, 'category'].str.lower()

# Whitespace issues
sample_idx = df_transactions.sample(frac=0.02).index
df_transactions.loc[sample_idx, 'payment_method'] = \
    ' ' + df_transactions.loc[sample_idx, 'payment_method'] + ' '

print("✅ Added missing values, duplicates, outliers")

# ============================================
# SAVE DATASETS
# ============================================

print("\nSaving datasets...")

# Shuffle transactions
df_transactions = df_transactions.sample(frac=1).reset_index(drop=True)

# Save
customers.to_csv('../data/customers.csv', index=False)
df_transactions.to_csv('../data/transactions_raw.csv', index=False)

print(f"✅ Saved customers.csv ({len(customers):,} rows)")
print(f"✅ Saved transactions_raw.csv ({len(df_transactions):,} rows)")

# ============================================
# DATASET SUMMARY
# ============================================

print("\n" + "="*70)
print("DATASET GENERATION COMPLETE!")
print("="*70)

print(f"""
Dataset Summary:
{'─'*70}
Customers:        {len(customers):,}
Transactions:     {len(df_transactions):,}
Date Range:       {df_transactions['transaction_date'].min().date()} to 
                  {df_transactions['transaction_date'].max().date()}
Total Revenue:    ₹{df_transactions['total_amount'].sum():,.2f}
Avg Transaction:  ₹{df_transactions['total_amount'].mean():,.2f}

Categories:       {df_transactions['category'].nunique()}
Products:         {df_transactions['product'].nunique()}
Cities:           {customers['city'].nunique()}

Data Quality Issues (Realistic):
{'─'*70}
✓ Missing values in rating and delivery_days
✓ Duplicate transactions (~1%)
✓ Outliers in total_amount
✓ Mixed case in categories
✓ Whitespace in payment methods

This messy data is PERFECT for demonstrating cleaning skills!
""")