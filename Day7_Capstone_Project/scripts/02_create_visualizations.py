"""
CAPSTONE PROJECT: Professional Visualizations
==============================================
Create publication-quality charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

print("="*70)
print("CREATING PROFESSIONAL VISUALIZATIONS")
print("="*70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load clean data
df = pd.read_csv('../data/transactions_clean.csv')
#Ensure transaction_datetime is datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Create time features (rebuild for this script)
df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
df['day_of_week'] = df['transaction_date'].dt.day_name()
df['month'] = df['transaction_date'].dt.month
df['month_name'] = df['transaction_date'].dt.month_name()
df['year'] = df['transaction_date'].dt.year

customers = pd.read_csv('../data/customers.csv')
df = df.merge(
    customers[['customer_id', 'customer_segment']],
    on='customer_id',
    how='left'
)

print(f"✅ Loaded {len(df):,} transactions")
print()

# VIZ 1: EXECUTIVE DASHBOARD

print("Creating Viz 1: Executive Dashboard...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('E-COMMERCE EXECUTIVE DASHBOARD', fontsize=22, fontweight='bold', y='0.98')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. Revenue Over Time
ax1 = fig.add_subplot(gs[0, :2])
monthly = df.groupby(df['transaction_date'].dt.to_period('M'))['total_amount'].sum()
monthly.index = monthly.index.to_timestamp()
ax1.plot(monthly.index, monthly.values, marker='o', linewidth=3, markersize=8, color='#2ecc71')
ax1.fill_between(monthly.index, monthly.values, alpha=0.3, color='#2ecc71')
ax1.set_title('Monthly Rent Trend', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Revenue (₹)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')
for i, v in enumerate(monthly.values):
    ax1.text(monthly.index[i], v, f'₹{v/1e6:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

#2. Key Mertics Card
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
total_revenue = df['total_amount'].sum()
total_trans = len(df)
avg_order = df['total_amount'].mean()
return_rate = df['returned'].mean() * 100

metrics_text = f"""
╔═══════════════════════════╗
║     KEY METRICS          ║
╠═══════════════════════════╣
║                          ║
║  Total Revenue:          ║
║  ₹{total_revenue/1e6:.2f} Million         ║
║                          ║
║  Transactions:           ║
║  {total_trans:,}                ║
║                          ║
║  Avg Order Value:        ║
║  ₹{avg_order:,.0f}                ║
║                          ║
║  Return Rate:            ║
║  {return_rate:.1f}%                  ║
║                          ║
╚═══════════════════════════╝
"""
ax2.text(0.5, 0.5, metrics_text, transform=ax2.transAxes,
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Category Revenue
ax3 = fig.add_subplot(gs[1, 0])
category_rev = df.groupby('category')['total_amount'].sum().sort_values(ascending=True)
colors = ['#3498db', '#e74c3c', '#f39c12']
ax3.barh(category_rev.index, category_rev.values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Revenue (₹)', fontsize=11, fontweight='bold')
ax3.set_title('Revenue by Category', fontsize=13, fontweight='bold', pad=10)
for i, v in enumerate(category_rev.values):
    ax3.text(v, i, f'₹{v/1e6:.2f}M', va='center', fontweight='bold', fontsize=10)

# 4. Customer Segments
ax4 = fig.add_subplot(gs[1, 1])
segment_rev = df.groupby(customers['customer_segment'])['total_amount'].sum()
colors_seg = ['#FFD700', '#C0C0C0', '#CD7F32']
wedges, texts, autotexts = ax4.pie(segment_rev.values, labels=segment_rev.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors_seg,
                                     explode=(0.05, 0, 0))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
ax4.set_title('Revenue by Segment', fontsize=13, fontweight='bold', pad=10)

# 5. Top 10 Cities
ax5 = fig.add_subplot(gs[1, 2])
city_rev = df.groupby(customers['city'])['total_amount'].sum().sort_values(ascending=False).head(10)
ax5.bar(range(len(city_rev)), city_rev.values, color='#9b59b6', edgecolor='black', linewidth=1.5)
ax5.set_xticks(range(len(city_rev)))
ax5.set_xticklabels(city_rev.index, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Revenue (₹)', fontsize=11, fontweight='bold')
ax5.set_title('Top 10 Cities', fontsize=13, fontweight='bold', pad=10)
ax5.grid(axis='y', alpha=0.3)

# 6. Payment Methods
ax6 = fig.add_subplot(gs[2, 0])
payment_counts = df['payment_method'].value_counts()
ax6.bar(payment_counts.index, payment_counts.values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
        edgecolor='black', linewidth=1.5)
ax6.set_ylabel('Transactions', fontsize=11, fontweight='bold')
ax6.set_title('Payment Method Distribution', fontsize=13, fontweight='bold', pad=10)
ax6.set_xticklabels(payment_counts.index, rotation=30, ha='right', fontsize=10)
ax6.grid(axis='y', alpha=0.3)

# 7. Ratings Distribution
ax7 = fig.add_subplot(gs[2, 1])
rating_counts = df['rating'].value_counts().sort_index()
colors_rating = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']
ax7.bar(rating_counts.index, rating_counts.values, color=colors_rating, edgecolor='black', linewidth=1.5)
ax7.set_xlabel('Rating (stars)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Count', fontsize=11, fontweight='bold')
ax7.set_title('Customer Ratings', fontsize=13, fontweight='bold', pad=10)
ax7.grid(axis='y', alpha=0.3)
avg_rating = df['rating'].mean()
ax7.axhline(y=rating_counts.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)

# 8. Device Usage
ax8 = fig.add_subplot(gs[2, 2])
device_counts = df['device'].value_counts()
sizes = device_counts.values
labels = [f"{l}\n{v:,}" for l, v in zip(device_counts.index, device_counts.values)]
colors_devices = ['#3498db', '#2ecc71', '#e74c3c']
ax8.pie(sizes, labels=labels, colors=colors_devices, autopct='%1.1f%%', startangle=90,
        textprops={'fontweight': 'bold', 'fontsize': 10})
ax8.set_title('Device type Usage', fontsize=13, fontweight='bold', pad=10)

plt.savefig('../visualizations/01_executive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_executive_dashboard.png")

# VIZ 2: CUSTOMER BEHAVIOUR ANALYSIS

print("Creating Viz 2: Customer Behaviour Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CUSTOMER BEHAVIOUR DEEP DIVE', fontsize=20, fontweight='bold')

# Customer Lifetime Value by Segment
customer_ltv = df.groupby(['customer_id', 'customer_segment'])['total_amount'].sum().reset_index()
segment_ltv = df.groupby('customer_segment')['total_amount'].mean()

axes[0, 0].bar(segment_ltv.index, segment_ltv.values,
               color=['#FFD700', '#3498db', '#95a5a6'], edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Avg LTV (₹)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Customer Lifetime Value by Segment', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(segment_ltv.values):
    axes[0, 0].text(i, v, f'₹{v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Purchase Frequency
purchase_freq = df.groupby('customer_id').size().value_counts().sort_index()
axes[0, 1].bar(purchase_freq.index, purchase_freq.values, color='#2ecc71', 
               edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Number of Purchases', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Purchase Frequency Distribution', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Avg Order Value by Segment Over Time
monthly_segment = df.groupby([df['transaction_date'].dt.to_period('M'), 'customer_segment'])['total_amount'].mean().unstack()
monthly_segment.index = monthly_segment.index.to_timestamp()
for col in monthly_segment.columns:
    axes[1, 0].plot(monthly_segment.index, monthly_segment[col],
                    marker='o', linewidth=2.5, label=col, markersize=7)
axes[1, 0].set_ylabel('Avg Order Value (₹)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Avg Order Value Trend by Segment', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

#Return Rate by Category
return_by_category = df.groupby('category')['returned'].mean()*100
axes[1, 1].barh(return_by_category.index, return_by_category.values, 
                color='#e74c3c', edgecolor='black', linewidth=2)
axes[1, 1].set_xlabel('Return Rate (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Return Rate by Category', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)
for i, v in enumerate(return_by_category.values):
    axes[1, 1].text(v, i, f' {v:.2f}%', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('../visualizations/02_customer_behaviour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_customer_behavior.png")

# VIZ 3: TIME-BASED PATTERNS
print("Creating Viz 3: Time-Based Patterns...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TIME-BASED PURCHASING PATTERNS', fontsize=20, fontweight='bold')

# Daily Revenue
daily_revenue = df.groupby(df['transaction_date'].dt.date)['total_amount'].sum()
axes[0, 0].plot(daily_revenue.index, daily_revenue.values, linewidth=1.5, color='#3498db', alpha=0.7)
axes[0, 0].fill_between(daily_revenue.index, daily_revenue.values, alpha=0.3, color='#3498db')
axes[0, 0].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Revenue (₹)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Daily Revenue Trend', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Day of Week Analysis
dow_revenue = df.groupby('day_of_week')['total_amount'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
colors_dow = ['#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#e74c3c', '#e74c3c']
axes[0, 1].bar(range(7), dow_revenue.values, color=colors_dow, edgecolor='black', linewidth=1.5)
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=10)
axes[0, 1].set_ylabel('Revenue (₹)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Revenue by Day of Week', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Monthly Comparison by Category
monthly_cat = df.groupby([df['transaction_date'].dt.to_period('M'),
                         'category'])['total_amount'].sum().unstack()
monthly_cat.index = monthly_cat.index.to_timestamp()
monthly_cat.plot(kind='area', stacked='True', ax=axes[1, 0],
                 color=['#3498db', '#2ecc71', '#f39c12'], alpha=0.7)
axes[1, 0].set_ylabel('Revenue (₹)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Monthly Revenue by Category (Stacked)', fontsize=14, fontweight='bold')
axes[1, 0].legend(title='Category', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Weekend vs Weekday
Weekend_comp = df.groupby('is_weekend')['total_amount'].agg(['sum', 'count', 'mean'])
Weekend_labels = ['Weekday', 'Weekend']
x = np.arange(2)
width = 0.35

ax = axes[1, 1]
ax2 = ax.twinx()

bars1 = ax.bar(x - width/2, Weekend_comp['sum'].values, width, label='Total Revenue',
               color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, Weekend_comp['mean'].values, width, label='Avg Order Value',
               color='#2ecc71', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Total Revenue (₹)', fontsize=12, fontweight='bold', color='#3498db')
ax2.set_ylabel('Avg Order Value (₹)', fontsize=12, fontweight='bold', color='#2ecc71')
ax.set_xlabel('Day Type', fontsize=12, fontweight='bold')
ax.set_title('Weekend vs Weekday Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(Weekend_labels)
ax.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/03_time_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_time_patterns.png")

# VIZ 4: GEOGRAPHIC & PRODUCT INSIGHTS

print("Creating Viz 4: Geographic & Product Analysis...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle('GEOGRAPHIC & PRODUCT PERFORMANCE', fontsize=20, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Top 15 Products
ax1 = fig.add_subplot(gs[0, :])
top_products = df.groupby('product')['total_amount'].sum().sort_values(ascending=False).head(15)
colors_products = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_products))) # plt.cm.viridis: Uses the "Viridis" color map (a perceptually uniform sequence from purple to yellow).
bars = ax1.barh(range(len(top_products)), top_products.values, color=colors_products,
                edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(top_products)))
ax1.set_yticklabels(top_products.index, fontsize=10)
ax1.set_xlabel('Revenue (₹)', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Products by Revenue', fontsize=15, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_products.values):
    ax1.text(v, i, f' ₹{v/1e6:.2f}M', va='center', fontweight='bold', fontsize=9)

# City Performance Map
ax2 = fig.add_subplot(gs[1, 0])
city_stats = df.groupby(customers['city']).agg({
    'total_amount': 'sum',
    'customer_id': 'nunique',
    'transaction_id': 'count'
}).sort_values('total_amount', ascending=False)

cities = city_stats.index[:8]
revenues = city_stats['total_amount'].values[:8]
customers = city_stats['customer_id'].values[:8]

ax2.bar(range(len(cities)), revenues, color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.set_xticks(range(len(cities)))
ax2.set_xticklabels(cities, rotation=45, ha='right', fontsize=10)
ax2.set_ylabel('Revenue (₹)', fontsize=12, fontweight='bold')
ax2.set_title('Top 8 Cities by Revenue', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Product Category Matrix
ax3 = fig.add_subplot(gs[1, 1])
category_product = df.groupby(['category', 'product'])['total_amount'].sum().unstack()
# Take top 5 products per category
top_prods_per_cat = []
for cat in category_product.index:
    top_5 = category_product.loc[cat].nlargest(3)
    for prod in top_5.index:
        top_prods_per_cat.append(prod)
top_prods_per_cat = list(set(top_prods_per_cat))[:10]

heatmap_data = category_product[top_prods_per_cat]
im = ax3.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(len(heatmap_data.columns)))
ax3.set_yticks(range(len(heatmap_data.index)))
ax3.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=9)
ax3.set_yticklabels(heatmap_data.index, fontsize=11)
ax3.set_title('Category × Product Revenue Heatmap', fontsize=14, fontweight='bold', pad=10)

# Add colorbar
cbar = fig.colorbar(im, ax=ax3)
cbar.set_label('Revenue (₹)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../visualizations/04_geographic_product.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 04_geographic_product.png")

print("\n" + "="*70)
print("ALL VISUALIZATIONS CREATED!")
print("="*70)
print(f"""
Created 4 comprehensive dashboards:
  ✅ 01_executive_dashboard.png     (8-panel overview)
  ✅ 02_customer_behavior.png       (Customer analysis)
  ✅ 03_time_patterns.png           (Temporal trends)
  ✅ 04_geographic_product.png      (Location & products)

Total charts: 20+ professional visualizations
All saved in: ../visualizations/
""")