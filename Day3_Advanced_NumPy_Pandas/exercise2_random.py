import numpy as np

# Task 1: Generate synthetic classification dataset
np.random.seed(42)

# Class 0: mean=[2, 2], std=1
class_0 = np.random.randn(100, 2) * 1.0 + np.array([2, 2])

# Class 1: mean=[5, 5], std=1.5
class_1 = np.random.randn(100, 2) * 1.5 + np.array([5, 5])

print("Class 0 data (first 5):\n", class_0[:5])
print("Class 1 data (first 5):\n", class_1[:5])

# Task 2: Create labels
labels_0 = np.zeros(100)
labels_1 = np.ones(100)

# Combine data
X = np.vstack([class_0, class_1])
y = np.concatenate([labels_0, labels_1])

print(f"Dataset Shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print()

#Task 3: Shuffle dataset
indicies = np.arange(len(X))
np.random.shuffle(indicies)

X_shuffled = X[indicies]
y_shuffled = y[indicies]

print("First 5 labels befor shuffle: ", y[:5])
print("First 5 labels after shuffle: ", y_shuffled[:5])
print()

# Task 4: Split into train (80%) and test (20%)
split = int(0.8 * len(X))
X_train, X_test = X_shuffled[:split], X_shuffled[split:]
y_train, y_test = y_shuffled[:split], y_shuffled[split:]

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
print()

#Task 5 : Initialized neural network weights
layer_size = [2,16,32,16,1] # Input -> Hidden -> Hidden -> Hidden -> Output
weights = []

for i in range(len(layer_size) - 1):
    n_in = layer_size[i]
    n_out = layer_size[i + 1]

    # Xavier initialization
    W = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
    weights.append(W)

    print(f"Layer {i+1} weights: {W.shape}, mean={np.mean(W):.6f}, std= {np.std(W):.6f}")