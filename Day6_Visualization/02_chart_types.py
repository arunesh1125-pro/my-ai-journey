"""
ESSENTIAL CHART TYPES FOR DATA SCIENCE
Each chart type answers specific questions!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("="*70)
print("ESSENTIAL CHART TYPES")
print("="*70)

# Sample data
np.random.seed(42)

# 1. SCATTER PLOT - Relationship between Variables

print("\n1. SCATTER PLOT (Correlation Analysis)")
print("-"*50)

# Generate correlated data
n = 200
x = np.random.randn(n)
y = 2 * x + 1 + np.random.randn(n) * 0.5

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, s=50, c='#3498db', edgecolors='black', linewidth=0.5)
plt.title('Scatter Plot: X vs Y (Positive Correlation)', fontsize=16, fontweight='bold')
plt.xlabel('X Variable', fontsize=12)
plt.ylabel('Y Variable', fontsize=12)
plt.grid(True, alpha=0.3)

# Add regression line
m, c = np.polyfit(x, y, 1) # perform least square polynomial fit to set of data points, reurns coeff of fitted polynomial
plt.plot(x, m*x + c, 'r--', linewidth=0.2, label=f'y = {m:2f}x + {c:.2f}')
plt.legend()

plt.savefig('05_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 05_scatter.png")

# 2. BAR CHART _ COmpare categories

print("\n2. BAR CHART (Category Comparison)")
print("-"*50)

categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title('Sales by Product', fontsize=16, fontweight='bold')
plt.xlabel('Product', fontsize=12)
plt.ylabel('Sales (in thousands)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('06_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 06_bar.png")

# 3. HISTOGRAM - Distribution of single variable

print("\n3. HISTOGRAM (Distribution Analysis)")
print("-"*50)

data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(data, bins=30, edgecolor='black', color='#3498db', alpha=0.7) # bins=30, 30 vertical bars, # patches= list of 30 "Rctangle" objects, where each object represents one physical bar in the plot

# Color bars by value
cm = plt.cm.RdYlGn_r # Sets up a color map that transitions from Red to Yellow to Green, reversed (_r), coloring lower-value bars differently than higher-value ones.
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(i/len(patches))) #,set_facecolor- updates the color for specific bar, cm- converts number into specific color, (i/len(cm))- NOrmalizes index to value b/ 0.0 and 1.0: eg: First bar: 0/30, mid bar: 15/30, last bar: 29/30

plt.axvline(data.mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {data.mean():.1f}')
plt.axvline(np.median(data), color='green', linestyle='--',
            linewidth=2, label=f'Median: {np.median(data):.1f}')

plt.title('Distribution of Test Scores', fontsize=16, fontweight='bold')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('07_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 07_histogram.png")

# 4. BOX PLOT - Statistical distribution

print("\n4. BOX PLOT (Statistical Summary)")
print("-"*50)

data1 = [np.random.normal(100, 10, 100),
         np.random.normal(90, 15, 100),
         np.random.normal(110, 12, 100),
         np.random.normal(85, 20, 100)]

plt.figure(figsize=(10, 6))
bp = plt.boxplot(data1, labels=['Group A', 'Group B', 'Group C', 'Group D'], 
                 patch_artist=True, notch=True, showmeans=True ) # patch_artist=True: Crucial for customization; it tells matplotlib to return filled patches (boxes) rather than just lines, allowing us to fill them with color.
                                                            # notch=True: Creates a notch in the box, which shows a confidence interval around the median.
                                                            # showmeans=True: Displays the mean of the data as a point (or marker).
# Customize box colors
colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for patch, color in zip(bp['boxes'], colors1):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('Performance Comparison Across Groups', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('08_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 08_boxplot.png")

# 5. PIE CHART - Parts of whole

print("\n5. PIE CHART (Proportion Analysis)")
print("-"*50)

labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [35, 30, 20, 15]
colors2 = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0) # Explode 1st slice # This "pops out" the first slice (Category A) by 10% of the radius to make it stand out.

plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, #wedges: actual slices of the pie chart, texts: lables for each size, autotext: percentage of text inside each slice
                                   colors=colors2, autopct='%1.1f%%',  #autopct='%1.1f%%': Automatically calculates and displays the percentage on each slice with one decimal point.
                                   shadow=True, startangle=90,  #Rotates the start of the first slice to the top (12 o'clock position) instead of the default 3 o'clock.
                                   textprops={'fontsize': 12})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

plt.title('Market Share Distribution', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')
plt.savefig('09_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 09_pie.png")

# 6. HEATMAP - Matrix visualization

print("\n6. HEATMAP (Correlation Matrix)")
print("-"*50)

# Generate correlation matrix
data2 = np.random.randn(5, 5)
corr = np.corrcoef(data2)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1) # correlation values from -1 to 1

# Labels
variables = ['Var A', 'Var B', 'Var C', 'Var D', 'Var E']
ax.set_xticks(np.arange(len(variables)))
ax.set_yticks(np.arange(len(variables)))
ax.set_xticklabels(variables)
ax.set_yticklabels(variables)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add correlation values
for i in range(len(variables)):
    for j in range(len(variables)):
        text = ax.text(j, i, f'{corr[i, j]:.2f}',
                       ha='center', va='center', color='black', fontweight='bold')
        
ax.set_title("Correlation Heatmap", fontsize=16, fontweight='bold')
fig.colorbar(im, ax=ax, label='Correlation Coefficient')
plt.tight_layout()
plt.savefig('10_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 10_heatmap.png")

# 7. AREA CHART - Change over time

print("\n7. AREA CHART (Trend Over Time)")
print("-"*50)

x1 = np.arange(0, 10, 0.1)
y1 = np.sin(x1) + 5
y2 = np.sin(x1 + 1) + 5
y3 = np.sin(x1 + 2) + 5

plt.figure(figsize=(12, 6))
plt.fill_between(x1, 0, y1, alpha=0.7, label='Product A', color='#e74c3c')
plt.fill_between(x1, y1, y1+y2, alpha=0.7, label='Product B', color='#3498db')
plt.fill_between(x1, y1+y2, y1+y2+y3, alpha=0.7, label='Product C', color='#2ecc71')

plt.title('Revenue Streams Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time (months)', fontsize=12)
plt.ylabel('Revenue ($1000s)', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('11_area.png', dpi=300,  bbox_inches='tight')
plt.close()
print("✅ Created: 11_area.png")

print("\n" + "="*70)
print("CHART TYPES COMPLETE!")
print("="*70)
print("""
When to use each chart:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scatter Plot  → Relationship between 2 numeric variables
Bar Chart     → Compare categories
Histogram     → Distribution of 1 numeric variable
Box Plot      → Statistical summary + outliers
Pie Chart     → Parts of whole (use sparingly!)
Heatmap       → Matrix/correlation data
Area Chart    → Trends over time, stacked comparison
Line Plot     → Trends over time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")