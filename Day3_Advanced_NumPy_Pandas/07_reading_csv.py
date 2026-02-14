import pandas as pd
import os

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#file_path = os.path.join(BASE_DIR, "sales_data.csv")

#df = pd.read_csv(file_path)

 # READING CSV FILES

print("="*60)
print("READING CSV DATA")
print("="*60)

# Read CSV
df = pd.read_csv('sales_data.csv')

print("Data loaded successfully!")
print(f"Shape: {df.shape}")
print()

# First look
print("First 5 rows: ")
print(df.head())
print()

print("Last 5 rows: ")
print(df.tail())
print()

# Info
print("DatFrame Info: ")
df.info()
print()

# Statistical summary
print("Statistical Summary: ")
print(df.describe())

 # DATA EXPLORATION

print("="*60)
print("DATA EXPLORATION")
print("="*60)

# Unique values
print("Unique products: ")
print(df['Product'].unique())
print()

print("Product counts: ")
print(df['Product'].value_counts())
print()

print("Unique regions: ")
print(df['Region'].unique())
print()

 # GROUPBY OPERATIONS

print("="*60)
print("GROUPBY (Like SQL GROUP BY)")
print("="*60)

# Average sales by product
print("Average sales by product: ")
product_avg = df.groupby('Product')['Sales'].mean()
print(product_avg)
print()

# Multiple aggregations
print("Sales statistics by product: ")
product_stats = df.groupby('Product')['Sales'].agg(['mean', 'sum', 'min', 'max', 'count'])
print(product_stats)
print()

 # FILTERING AND TRANSFORMATION

print("="*60)
print("FILTERING")
print("="*60)


# High-value sales (>50000)
high_value = df[df['Sales'] > 50000]
print(f"High-value sales (>50000): {len(high_value)} records")
print(high_value.head())
print()

# Specific Product
laptops = df[df['Product']== 'Laptop']
print(f"Laptop sales: {len(laptops)} records")
print(f"Total laptop revenue: ₹{laptops['Sales'].sum():,}")
print()

# Top-rated products
top_rated = df[df['Customer Rating'] >= 4.5]
print(f"Top-rated products (>=4.5): {len(top_rated)} records")
print(top_rated[['Product', 'Customer Rating', 'Sales']].head(10))
print()

 # SAVING PROCESSED DATA

print("="*60)
print("SAVING DATA")
print("="*60)

# Filter and save
high_value_sales = df[df['Sales']> 52000]
high_value_sales.to_csv('high_value_sales.csv', index=False)
print("✅ Saved high_value_sales.csv")

# Save with specific colums
df[['Date', 'Product', 'Sales']].to_csv('sales_summary.csv', index=False)