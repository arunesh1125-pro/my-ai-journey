"""
CONVOLUTIONAL NEURAL NETWORKS (CNNs)
====================================
The architecture that revolutionized computer vision
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

print("="*80)
print("CONVOLUTIONAL NEURAL NETWORKS (CNNs)")
print("="*80)

# ============================================
# WHY CNNs FOR IMAGES?
# ============================================

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                  WHY CNNs? THE IMAGE PROBLEM                           ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  YESTERDAY'S APPROACH (Dense/Fully Connected):                         ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Fashion MNIST: 28×28 = 784 pixels                              │ ║
║  │                                                                  │ ║
║  │  Input Layer:    784 neurons (FLATTENED image)                  │ ║
║  │  Hidden Layer:   128 neurons                                    │ ║
║  │  Connections:    784 × 128 = 100,352 weights!                   │ ║
║  │                                                                  │ ║
║  │  PROBLEMS:                                                       │ ║
║  │  ❌ Lost spatial information (flattened image)                  │ ║
║  │  ❌ Too many parameters (100K+ for tiny 28×28 image!)          │ ║
║  │  ❌ No translation invariance (cat in corner ≠ cat in center)  │ ║
║  │  ❌ Doesn't scale (200×200 = 40K inputs = millions of params!) │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  CNN APPROACH (Designed for Images):                                   ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Fashion MNIST: 28×28×1 (keep 2D structure!)                    │ ║
║  │                                                                  │ ║
║  │  Conv Layer:     3×3 filters (9 weights per filter)             │ ║
║  │  Connections:    Sparse! Each neuron sees only 3×3 region       │ ║
║  │  Parameters:     WAY fewer (thousands vs millions)              │ ║
║  │                                                                  │ ║
║  │  BENEFITS:                                                       │ ║
║  │  ✅ Preserves spatial structure (2D → 2D)                       │ ║
║  │  ✅ Fewer parameters (parameter sharing)                        │ ║
║  │  ✅ Translation invariance (same filter everywhere)             │ ║
║  │  ✅ Learns hierarchical features (edges → textures → objects)   │ ║
║  │  ✅ SCALES to large images (same filter size!)                  │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# ============================================
# WHAT IS CONVOLUTION?
# ============================================

print("\n" + "="*80)
print("CONVOLUTION OPERATION: THE CORE OF CNNs")
print("="*80)

print("""
CONVOLUTION = Sliding a filter/kernel over an image
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE: Edge Detection

Input Image (5×5):          Filter/Kernel (3×3):
┌─────────────────┐         ┌───────────┐
│ 0  0  0  0  0  │         │ -1  -1  -1 │  ← Detects
│ 0  0  0  0  0  │         │  0   0   0 │    horizontal
│ 1  1  1  1  1  │    ✱    │  1   1   1 │    edges
│ 1  1  1  1  1  │         └───────────┘
│ 1  1  1  1  1  │
└─────────────────┘

Convolution Process:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Place filter on top-left 3×3 region
┌───────────┐
│ 0  0  0  │ ·  │ -1  -1  -1 │
│ 0  0  0  │ ✱  │  0   0   0 │
│ 1  1  1  │    │  1   1   1 │
└───────────┘

Calculate: (0×-1) + (0×-1) + (0×-1) +
           (0×0)  + (0×0)  + (0×0)  +
           (1×1)  + (1×1)  + (1×1)  = 3

Output[0,0] = 3 ✅

Step 2: Slide filter RIGHT by 1 pixel (stride=1)
    ┌───────────┐
│ 0  0  0  0 │ ·  │ -1  -1  -1 │
│ 0  0  0  0 │ ✱  │  0   0   0 │
│ 1  1  1  1 │    │  1   1   1 │
    └───────────┘

Calculate: (0×-1) + (0×-1) + (0×-1) +
           (0×0)  + (0×0)  + (0×0)  +
           (1×1)  + (1×1)  + (1×1)  = 3

Output[0,1] = 3 ✅

Step 3: Continue sliding across and down...

Final Output (3×3):        ← Smaller than input!
┌───────────┐
│ 3  3  3  │  ← Edge detected here!
│ 3  3  3  │
│ 0  0  0  │
└───────────┘

The bright line at the edge shows where the edge is! 🎯


KEY CONCEPTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- FILTER/KERNEL: Small matrix (usually 3×3 or 5×5)
  - Learned during training! (not hand-designed)
  - Each filter detects one pattern (edge, corner, texture)

- STRIDE: How many pixels to move filter
  - Stride=1: Move 1 pixel at a time (more detail)
  - Stride=2: Move 2 pixels (faster, less detail)

- PADDING: Add border of zeros around image
  - 'valid': No padding (output smaller)
  - 'same': Padding added (output same size as input)

- FEATURE MAP: Output after convolution
  - Each filter produces one feature map
  - 32 filters → 32 feature maps


MATHEMATICAL FORMULA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output[i,j] = Σ Σ Input[i+m, j+n] × Filter[m,n] + bias
              m n

Where:
- (i,j) = position in output
- (m,n) = position in filter
- Σ Σ = sum over all filter positions


OUTPUT SIZE CALCULATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output_size = (Input_size - Filter_size + 2×Padding) / Stride + 1

Example:
Input: 28×28
Filter: 3×3
Padding: 0 (valid)
Stride: 1

Output = (28 - 3 + 0) / 1 + 1 = 26×26

With 'same' padding:
Padding = 1 (automatically calculated)
Output = (28 - 3 + 2) / 1 + 1 = 28×28 (same as input!)
""")

# Visual demonstration
def convolution_demo():
    """Demonstrate convolution operation"""
    
    # Create simple image with edge
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ], dtype=np.float32)
    
    # Edge detection filter
    filter_horizontal = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    
    # Vertical edge filter
    filter_vertical = np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ], dtype=np.float32)
    
    # Manual convolution (for demonstration)
    def manual_convolve(image, kernel):
        output_size = image.shape[0] - kernel.shape[0] + 1
        output = np.zeros((output_size, output_size))
        
        for i in range(output_size):
            for j in range(output_size):
                region = image[i:i+3, j:j+3]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    output_h = manual_convolve(image, filter_horizontal)
    output_v = manual_convolve(image, filter_vertical)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Horizontal edge detection
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Input Image\n(5×5)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    for i in range(5):
        for j in range(5):
            axes[0, 0].text(j, i, f'{int(image[i,j])}', 
                           ha='center', va='center', color='red', fontsize=11)
    
    axes[0, 1].imshow(filter_horizontal, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title('Horizontal Edge Filter\n(3×3)', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    for i in range(3):
        for j in range(3):
            axes[0, 1].text(j, i, f'{int(filter_horizontal[i,j])}', 
                           ha='center', va='center', fontsize=11, fontweight='bold')
    
    im = axes[0, 2].imshow(output_h, cmap='hot', vmin=-3, vmax=3)
    axes[0, 2].set_title('Output Feature Map\n(3×3) - Edge Detected!', 
                        fontweight='bold', fontsize=12, color='green')
    axes[0, 2].axis('off')
    for i in range(3):
        for j in range(3):
            axes[0, 2].text(j, i, f'{int(output_h[i,j])}', 
                           ha='center', va='center', color='white', 
                           fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Row 2: Vertical edge detection
    axes[1, 0].imshow(image.T, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Input Image (Rotated)\n(5×5)', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(filter_vertical, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('Vertical Edge Filter\n(3×3)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{int(filter_vertical[i,j])}', 
                           ha='center', va='center', fontsize=11, fontweight='bold')
    
    output_v_rotated = manual_convolve(image.T, filter_vertical)
    im2 = axes[1, 2].imshow(output_v_rotated, cmap='hot', vmin=-3, vmax=3)
    axes[1, 2].set_title('Output Feature Map\n(3×3) - Vertical Edge!', 
                        fontweight='bold', fontsize=12, color='green')
    axes[1, 2].axis('off')
    for i in range(3):
        for j in range(3):
            axes[1, 2].text(j, i, f'{int(output_v_rotated[i,j])}', 
                           ha='center', va='center', color='white', 
                           fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 2])
    
    plt.suptitle('CONVOLUTION OPERATION DEMONSTRATION', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('01_convolution_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 01_convolution_demo.png")

convolution_demo()

# ============================================
# POOLING LAYERS
# ============================================

print("\n" + "="*80)
print("POOLING LAYERS: DOWNSAMPLING")
print("="*80)

print("""
POOLING = Reduce spatial dimensions (make image smaller)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY POOLING?
✅ Reduce number of parameters (prevent overfitting)
✅ Reduce computation (faster training)
✅ Provide translation invariance (slight shifts OK)
✅ Extract dominant features


MAX POOLING (Most Common):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input (4×4):               Output (2×2):
┌──────────────┐           ┌────────┐
│ 1  3│ 2  4 │           │ 3 │ 4  │
│ 2  2│ 1  3 │    →      ├───┼────┤
├─────┼──────┤            │ 9 │ 7  │
│ 5  9│ 6  7 │           └────────┘
│ 1  8│ 4  2 │
└──────────────┘

Process (2×2 filter, stride=2):
1. Top-left region: [1,3,2,2] → max = 3
2. Top-right region: [2,4,1,3] → max = 4
3. Bottom-left region: [5,9,1,8] → max = 9
4. Bottom-right region: [6,7,4,2] → max = 7

Takes the STRONGEST activation in each region!


AVERAGE POOLING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Same input:                Output (2×2):
┌──────────────┐           ┌────────┐
│ 1  3│ 2  4 │           │2.0│2.5 │
│ 2  2│ 1  3 │    →      ├───┼────┤
├─────┼──────┤            │5.8│4.8 │
│ 5  9│ 6  7 │           └────────┘
│ 1  8│ 4  2 │
└──────────────┘

Takes the AVERAGE activation in each region.

MaxPooling is preferred (keeps strongest features).


TYPICAL CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layers.MaxPooling2D(pool_size=(2, 2))

- Pool size: 2×2 (most common)
- Stride: Defaults to pool_size (non-overlapping)
- Result: Halves height and width (4×4 → 2×2)
""")

# Pooling demonstration
def pooling_demo():
    """Demonstrate pooling operation"""
    
    # Create sample feature map
    feature_map = np.array([
        [1, 3, 2, 4],
        [2, 2, 1, 3],
        [5, 9, 6, 7],
        [1, 8, 4, 2]
    ], dtype=np.float32)
    
    # Max pooling manually
    def max_pool_2x2(image):
        output = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                region = image[i*2:(i+1)*2, j*2:(j+1)*2]
                output[i, j] = np.max(region)
        return output
    
    # Average pooling manually
    def avg_pool_2x2(image):
        output = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                region = image[i*2:(i+1)*2, j*2:(j+1)*2]
                output[i, j] = np.mean(region)
        return output
    
    max_output = max_pool_2x2(feature_map)
    avg_output = avg_pool_2x2(feature_map)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input
    im1 = axes[0].imshow(feature_map, cmap='viridis', vmin=0, vmax=9)
    axes[0].set_title('Input Feature Map\n(4×4)', fontweight='bold', fontsize=13)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{int(feature_map[i,j])}', 
                        ha='center', va='center', color='white', 
                        fontsize=12, fontweight='bold')
    # Draw grid
    axes[0].axvline(x=1.5, color='red', linewidth=3)
    axes[0].axhline(y=1.5, color='red', linewidth=3)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0])
    
    # Max pooling output
    im2 = axes[1].imshow(max_output, cmap='viridis', vmin=0, vmax=9)
    axes[1].set_title('Max Pooling Output\n(2×2)', fontweight='bold', fontsize=13, color='green')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{int(max_output[i,j])}', 
                        ha='center', va='center', color='white', 
                        fontsize=14, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1])
    
    # Average pooling output
    im3 = axes[2].imshow(avg_output, cmap='viridis', vmin=0, vmax=9)
    axes[2].set_title('Average Pooling Output\n(2×2)', fontweight='bold', fontsize=13)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f'{avg_output[i,j]:.1f}', 
                        ha='center', va='center', color='white', 
                        fontsize=14, fontweight='bold')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle('POOLING OPERATION DEMONSTRATION', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('01_pooling_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 01_pooling_demo.png")

pooling_demo()

# ============================================
# CNN ARCHITECTURE
# ============================================

print("\n" + "="*80)
print("TYPICAL CNN ARCHITECTURE")
print("="*80)

print("""
STANDARD CNN PATTERN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CONV → ReLU → POOL] × N → [FC → ReLU] × M → FC → Softmax

Where:
- CONV = Convolutional layer
- ReLU = Activation function
- POOL = Pooling layer (MaxPool)
- FC = Fully Connected (Dense) layer
- N = Number of conv blocks (typically 2-5)
- M = Number of FC layers (typically 1-2)


EXAMPLE: Simple CNN for 28×28 Images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:           28×28×1    (grayscale image)
                    ↓
Conv2D(32):      28×28×32   (32 filters, 3×3, padding='same')
                    ↓
ReLU:            28×28×32   (activation)
                    ↓
MaxPool2D:       14×14×32   (2×2 pool → halves size)
                    ↓
Conv2D(64):      14×14×64   (64 filters, 3×3, padding='same')
                    ↓
ReLU:            14×14×64
                    ↓
MaxPool2D:       7×7×64     (2×2 pool → halves size)
                    ↓
Flatten:         3,136      (7×7×64 = 3,136 values)
                    ↓
Dense(128):      128        (fully connected layer)
                    ↓
ReLU:            128
                    ↓
Dropout(0.5):    128        (prevent overfitting)
                    ↓
Dense(10):       10         (output layer, 10 classes)
                    ↓
Softmax:         10         (probabilities)


KERAS CODE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                  input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])


PARAMETER COUNT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conv2D(32, 3×3):  (3×3×1 + 1) × 32 = 320 params
  (filter_size × input_channels + bias) × num_filters

Conv2D(64, 3×3):  (3×3×32 + 1) × 64 = 18,496 params

Dense(128):       3,136 × 128 + 128 = 401,536 params
Dense(10):        128 × 10 + 10 = 1,290 params

TOTAL: ~421K params

Compare to Dense network (Day 16):
Input(784) → Dense(128) = 784×128 = 100,352 params for ONE layer!

CNNs have FEWER parameters despite being more powerful! ✅
""")

# Build example CNN
print("\n🔨 Building Example CNN:")

model_example = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                  input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("✅ CNN model created!\n")
print("📊 Model Summary:")
model_example.summary()

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

insights = """
🎓 WHY CNNs REVOLUTIONIZED COMPUTER VISION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PARAMETER SHARING
   Same filter applied to entire image
   → Dramatically fewer parameters than Dense layers
   → 3×3 filter = 9 weights, but applied to 28×28 = 784 locations!

2. TRANSLATION INVARIANCE
   Cat detected anywhere in image (not just specific position)
   → Same filter everywhere = recognizes patterns regardless of location

3. HIERARCHICAL FEATURE LEARNING
   Layer 1: Edges & simple textures (horizontal, vertical, diagonal)
   Layer 2: Corners & curves (combining edges)
   Layer 3: Object parts (eyes, wheels, windows)
   Layer 4: Full objects (faces, cars, buildings)
   
   Each layer builds on previous! 🧱

4. SPATIAL STRUCTURE PRESERVED
   Dense layers: Flatten image → lose structure
   CNNs: Keep 2D structure → preserve spatial relationships
   
   Nearby pixels are related! (nose near eyes, wheels under car)

5. SCALES TO LARGE IMAGES
   3×3 filter works for 28×28 OR 1024×1024!
   Dense layer: 1024×1024 = 1M inputs = MILLIONS of parameters!


📊 COMPARISON: Dense vs CNN on Images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

28×28 Image (Fashion MNIST):

Dense Network:
- Input: 784 (flattened)
- Structure lost: ❌
- Parameters: ~100K
- Accuracy: 88.5%

CNN:
- Input: 28×28×1 (2D preserved)
- Structure kept: ✅
- Parameters: ~421K (but more efficient!)
- Accuracy: 91%+ (we'll see tomorrow!)

Winner: CNN! 🏆


🎯 PRACTICAL TIPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Start with 32-64 filters in first Conv layer
✅ Double filters after each pooling (32 → 64 → 128)
✅ Use 3×3 filters (most common, works great)
✅ Use 'same' padding to keep dimensions
✅ Always: Conv → Activation → Pool
✅ Use MaxPooling (better than Average for most cases)
✅ End with 1-2 Dense layers before output
"""

print(insights)

print("\n" + "="*80)
print("SESSION 1 COMPLETE: CNN Theory Understood!")
print("="*80)
print("\n☕ Take a 15-minute break before building CIFAR-10 classifier!")