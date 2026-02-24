"""
MATPLOTLIB: The Foundation of Python Visualization
Every other plotting library builds on Matplotlib!
"""

import matplotlib.pyplot as plt
import numpy as np


print("="*70)
print("MATPLOTLIB FUNDAMENTALS")
print("="*70)

# ANATONMY OF A MATPLOTLIB FIGURE

print("""
MATPLOTLIB STRUCTURE:
┌─────────────────────────────────┐
│ Figure                          │  ← Container for everything
│  ┌─────────────────────────┐   │
│  │ Axes (Plot Area)        │   │  ← Where data is plotted
│  │  ┌─────────────────┐   │   │
│  │  │ Data            │   │   │  ← Your actual data
│  │  └─────────────────┘   │   │
│  │  Title, Labels, etc.   │   │  ← Annotations
│  └─────────────────────────┘   │
└─────────────────────────────────┘

Key terms:
- Figure: The entire visualization (can have multiple plots)
- Axes: Individual plot area (not same as axis!)
- Axis: The x-axis or y-axis lines
- Artist: Everything you can see (lines, text, etc.)
""")

# BASIC LINE PLOT

print("\n1. BASIC LINE PLOT")
print("-"*50)

# Data 
x = np.linspace(0, 10, 100) # linspace = return evenly spaced values, (start, stop, n values(10) - 10 evenky spaced values from start to stop)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6)) # Width=10, height=6 # 1. Figure
plt.plot(x, y) # 2. Plot axes(2 individual axis)
plt.title('Sin Wave') # Annotation
plt.xlabel('X axis')  # Annotation in axis
plt.ylabel('Y axis')
plt.grid(True, alpha=0.3)  # Anotation in plot
plt.savefig('01_basic_line.png', dpi=300, bbox_inches='tight') # Save the Graph chart
plt.close()

print("✅ Created: 01_basic_line.png")

# MULTIPLE LINES

print("\n2. MULTIPLE LINES ON SAME PLOT")
print("-"*50)

x1 = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.exp(-x/10)

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label='sin(x)', linewidth=2)
plt.plot(x1, y2, label='cos(x)', linewidth=2)
plt.plot(x1, y3, label='sin(x) * e^(-x/10)', linewidth=2)

plt.title('Multiple Functions', fontsize=16, fontweight='bold')
plt.xlabel('X axis', fontsize=12)
plt.ylabel('Y axis', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('02_multiple_lines.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created: 02_multiple_lines.png")

# LINE STYLES AND MARKERS

print("\n3. LINE STYLES AND MARKERS")
print("-"*50)

x2 = np.linspace(0, 10, 20)

plt.figure(figsize=(12, 8))

# Different line styles
plt.subplot(2, 2, 1) #nrows, ncolumns, index
plt.plot(x, x**2, '-', label='solid')
plt.plot(x, x**1.8, '--', label='dashed')
plt.plot(x, x**1.6, '-.', label='dashdot')
plt.plot(x, x**1.4, ':', label='dotted')
plt.title('Line Styles')
plt.legend()
plt.grid(True, alpha=0.3)

# Different markers
plt.subplot(2, 2, 2)
plt.plot(x, x**2, 'o-', label='circle')
plt.plot(x, x**1.8, 's-', label='square')
plt.plot(x, x**1.6, '^-', label='triangle')
plt.plot(x, x**1.4, 'D-', label='diamond')
plt.title('Markers')
plt.legend()
plt.grid(True, alpha=0.3)

# Colors
plt.subplot(2, 2, 3)
plt.plot(x, x**2, 'r-', label='red')
plt.plot(x, x**1.8, 'g-', label='green')
plt.plot(x, x**1.6, 'b-', label='blue')
plt.plot(x, x**1.4, 'k-', label='black')
plt.title('Colors (Single Letter)')
plt.legend()
plt.grid(True, alpha=0.3)

# Hex colors
plt.subplot(2, 2, 4)
plt.plot(x, x**2, color='#FF6B6B', linewidth=3, label='Coral')
plt.plot(x, x**1.8, color='#4ECDC4', linewidth=3, label='Turquoise')
plt.plot(x, x**1.6, color='#45B7D1', linewidth=3, label='Sky Blue')
plt.plot(x, x**1.4, color='#FFA07A', linewidth=3, label='Light Salmon')
plt.title('Hex Colors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_styles_markers.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created: 03_styles_markers.png")

# CUSTOMIZATION

print("\n4. PROFESSIONAL CUSTOMIZATION")
print("-"*50)

a = np.linspace(0, 2*np.pi, 100)
b = np.sin(a)

plt.figure(figsize=(12, 6))

plt.plot(a, b, color='#2E86AB', linewidth=3, label='sin(x)')
plt.fill_between(a, 0, b, alpha=0.3, color='#A23B72')

plt.title('Professional Sine Wave',
          fontsize=18,
          fontweight='bold',
          pad=20)
plt.xlabel('Angle (radians)', fontsize=14, fontweight='bold')
plt.ylabel('Amplitutde', fontsize=14, fontweight='bold')

# Custom ticks
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize=12)
plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=12)

#reference lines
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=np.pi, color='r', linestyle='--', linewidth=1, alpha=0.5, label='π')

#Legend
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

# Style the axes
ax = plt.gca()  # get the current axes instance
ax.spines['top'].set_visible(False) # Hide the top border
ax.spines['right'].set_visible(False)

plt.savefig('04_professional.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created: 04_professional.png")

print("\n" + "="*70)
print("MATPLOTLIB BASICS COMPLETE!")
print("="*70)