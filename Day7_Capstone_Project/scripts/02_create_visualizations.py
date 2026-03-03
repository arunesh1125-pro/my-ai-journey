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
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
customers = pd.read_csv('../data/customers.csv')

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