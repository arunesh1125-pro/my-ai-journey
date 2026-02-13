"""
MINI PROJECT: Sales Data Analysis
Comprehensive analysis of sales dataset
"""

import pandas as pd
import numpy as np

print("="*70)
print("SALES DATA ANALYSIS REPORT")
print("="*70)
print()

 # 1. LOAD DATA

df = pd.read_csv('sales_data.csv')
print("📊 DATA LOADED ")
print(f"Total record: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

 # 2. OVERVIEW STATISTICS

print("="*70)
print("SECTION 1: OVERVIEW STATISTICS")
print("="*70)

total_revenue = df['Sales'].sum()
total_quantity = df['Quantity'].sum()
avg_sale = df['Sales'].mean()
avg_rating = df['Customer Rating'].mean()

print(f" Total Revenue: ₹{total_revenue:,}")
print(f" Total Quantity Sold: {total_quantity:,}")
print(f" Average Sale Value: ₹{avg_sale:,.2f}")
print(f" Average Customer Rating: {avg_rating:.2f}/5.0")

# 3. PRODUCT ANALYSIS

print("="*70)
print("SECTION 2: PRODUCT-WISE ANALYSIS")
print("="*70)

product_analysis = df.groupby('Product').agg({
    'Sales': ['sum','mean','count'],
    'Quantity': 'sum',
    'Customer Rating': 'mean'
}).round(2)

product_analysis.columns = ['Total_Revenue', 'Avg_Sale', 'Num_Sales', 'Total_Qty', 'Avg_Rating']
product_analysis = product_analysis.sort_values('Total_Revenue', ascending=False)

print("\nProduct Performance: ")
print(product_analysis)
print()

# Best and worst products
best_products = product_analysis['Total_Revenue'].idxmax()
worst_products = product_analysis['Total_Revenue'].idxmin()

print(f"Best Selling Product: {best_products}")
print(f"  Revenue: ₹{product_analysis.loc[best_products, 'Total_Revenue']:,.2f}")
print()

print(f" Lowest Selling Product: {worst_products}")
print(f" Revenue: ₹{product_analysis.loc[worst_products, 'Total_Revenue']:,.2f}")
print()

# 4. REGION ANALYSIS

print("="*70)
print("SECTION 3: REGION-WISE ANALYSIS")
print("="*70)

region_analysis = df.groupby('Region').agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Customer Rating': 'mean'
}).round(2)

region_analysis = region_analysis.sort_values('Sales', ascending= False)

print("\nRegion Performance: ")
print(region_analysis)
print()

best_region = region_analysis['Sales'].idxmax()
print(f" Top Performing Region: {best_region}")
print(f"  Revenue: ₹{region_analysis.loc[best_region, 'Sales']:,.2f}")
print()

# 5. PRODUCT x REGION INSIGHTS

pivot = df.pivot_table(
    values='Sales',
    index='Product',
    columns='Region',
    aggfunc='sum',
    fill_value=0
).round(2)

print("\nSales by Product and Region: ")
print(pivot)
print()

# 6. RATING ANALYSIS

print("="*70)
print("SECTION 5: CUSTOMER SATISFACTION")
print("="*70)

rating_distribution = df.groupby('Customer Rating').size()
print("\nRating Distribution: ")
print(rating_distribution)
print()

high_rated = df[df['Customer Rating'] >= 4.5]
print(f"High-rated transactions (>=4.5): {len(high_rated)} ({len(high_rated)/len(df)*100:.1f}%)")
print()

# Products with highest average rating
product_ratings = df.groupby('Product')['Customer Rating'].mean().sort_values(ascending=False)
print("Products by Average Rating: ")
print(product_ratings)
print()

 # 7. KEY INSIGHTS & RECOMMENDATIONS

print("="*70)
print("SECTION 6: KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n📌 KEY FINDINGS:")
print(f"1. {best_products} is the top-selling product (₹{product_analysis.loc[best_products, 'Total_Revenue']:,.2f})")
print(f"2. {best_region} is the most profitable region")
print(f"3. Overall customer satisfication: {avg_rating:.2f}/5.0")

# Identify Underperforming combinations
print("\n⚠️  AREAS FOR IMPROVEMENT:")

# Products with low sales
low_sales_threshold = df.groupby('Product')['Sales'].sum().quantile(0.25)
low_performing = product_analysis[product_analysis['Total_Revenue'] < low_sales_threshold]

#low-rated products
low_rated = df.groupby('Product')['Customer Rating'].mean()
low_rated = low_rated[low_rated < 4.0]
if len(low_performing) > 0:
    print(f"Products with low rating (<4.0): {', '.join(low_rated.index.tolist())}")

print("\n💡 RECOMMENDATIONS:")
print("1. Increase inventory for top-selling products")
print(f"2. Focus marketing efforts in {best_region} region")
print("3. Investigate and improve low-rated products")
print("4. Consider regional preferences for product placement")

# 8. EXPORT SUMMARY

print("\n" + "="*70)
print("EXPORTING ANALYSIS RESULTS")
print("="*70)

# Save summary
product_analysis.to_csv('product_summary.csv')
region_analysis.to_csv('regional_summary.csv')
pivot.to_csv('product_region_matrix.csv')

print("✅ Exported product_summary.csv")
print("✅ Exported region_summary.csv")
print("✅ Exported product_region_matrix.csv")
print()

print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
