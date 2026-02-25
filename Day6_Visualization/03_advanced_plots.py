"""
ADVANCED MATPLOTLIB TECHNIQUES
Multi-panel plots, subplots, and complex layouts
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

print("="*70)
print("ADVANCED PLOTTING TECHNIQUES")
print("="*70)

np.random.seed(42)

# 1. SUB PLOTS - Multiple plots in grid

print("\n1. SUBPLOTS (Grid Layout)")
print("-"*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Multiple Subplots Dashboard', fontsize=18, fontweight='bold')

# Plot 1: Line
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sin Wave')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
axes[0, 1].set_title('Rnadom Scatter')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bar
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[0, 2].bar(categories, values, color='green', alpha=0.7)
axes[0, 2].set_title('Category Comparison')

# Plot 4: Histogram
data = np.random.normal(100, 15, 100)
axes[1, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution')

# Plot 5: Box plot
box_data = [np.random.normal(100, 10, 100) for _ in range(4)]
axes[1, 1].boxplot(box_data, labels=['G1', 'G2', 'G3', 'G4'])
axes[1, 1].set_title('Group Comparison')

# Plot 6: Pie
sizes = [35, 30, 20, 15]
axes[1, 2].pie(sizes, labels=['A', 'B', 'C', 'D'], autopct='%1.1f%%')
axes[1, 2].set_title('Proportions')

plt.tight_layout()
plt.savefig('12_subplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 12_subplots.png")

# 2. GRIDSPEC - Custom subplot layouts

print("\n2. GRIDSPEC (Custom Layouts)")
print("-"*50)

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Large plot spanning 2 rows, 2 columns
ax1 = fig.add_subplot(gs[0:2, 0:2])
x = np.linspace(0, 2*np.pi, 100)
ax1.plot(x, np.sin(x), linewidth=3, color='#e74c3c')
ax1.set_title('Main Plot (Large)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Small plot top right
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(['A', 'B','C'], [10, 20, 15], color='#3498db')
ax2.set_title('Small 1')

# Small plot middle right
ax3 = fig.add_subplot(gs[1, 2])
ax3.scatter(np.random.randn(50), np.random.randn(50), alpha=0.6, color='#2ecc71')
ax3.set_title('Small 2')

# Bottom spanning plot
ax4 = fig.add_subplot(gs[2, :])
time = np.arange(100)
ax4.fill_between(time, np.sin(time/10), alpha=0.5, color='#9b59b6')
ax4.set_title('Bottom Plot (Full Width)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('13_gridspec.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 13_gridspec.png")

# TWIN AXES - Two Y - axes

print("\n3. TWIN AXES (Two Y-axes)")
print("-"*50)

fig, ax5 = plt.subplots(figsize=(12, 6))

x1 = np.arange(0, 10, 0.1)
y1 = np.exp(x1/3) # Exponential growth
y2 = np.sin(2*x1) * 10

# First y-axis
color1='#e74c3c'
ax5.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Revenue ($)', color=color1, fontsize=12, fontweight='bold')
ax5.plot(x1, y1, color=color1, linewidth=3, label='Revenue')
ax5.tick_params(axis='y', labelcolor=color1)
ax5.grid(True, alpha=0.3)


# Second y-axis
ax6 = ax5.twinx()
color2 = '#3498db'
ax6.set_ylabel('Customer Satisfication', color=color2, fontsize=12, fontweight='bold')
ax6.plot(x1, y2, color=color2, linewidth=3, linestyle='--', label='Satisfication')
ax6.tick_params(axis='y', labelcolor=color2)

plt.title('Revenue vs Customer Satisfication', fontsize=16, fontweight='bold', pad=20)

# combine legends
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax6.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()
plt.savefig('14_twin_axes.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 14_twin_axes.png")

# 4. ANNOTATIONS

print("\n4. ANNOTATIONS (Highlighting Key Points)")
print("-"*50)

fig, ax7 = plt.subplots(figsize=(12, 6))

a = np.linspace(0, 10, 100)
b = np.sin(a) * np.exp(-a/10)

ax7.plot(a, b, linewidth=3, color='#2c3e50')

# Final peaks
peaks_x = [np.pi/2, 5*np.pi/2]
peaks_y = [np.sin(px) * np.exp(px) for px in peaks_x]

# Annotate peaks
for px, py in zip(peaks_x, peaks_y):
    ax7.annotate(f'Peak\n({px:.2f}, {py:.2f})',
                 xy=(px, py),
                 xytext=(px+1, py+0.1),
                 fontsize=12,
                 fontweight='bold',
                 bbox=dict(boxstyle='round, pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3',
                                 color='red', lw=2))

# Add horizontal line
ax7.axhline(y=0, color='k', linestyle='--', linewidth=1,  alpha=0.3)

# Add shaded region
ax7.axvspan(6, 8, alpha=0.2, color='green', label='Important Region')

ax7.set_title('Annotated Plot with Key Insights', fontsize=16, fontweight='bold')
ax7.set_xlabel('X', fontsize=12)
ax7.set_ylabel('Y', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.savefig('15_annotations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Created: 15_annotations.png")

print("\n" + "="*70)
print("ADVANCED PLOTS COMPLETE!")
print("="*70)