import numpy as np

"""
Random numbers are CRITICAL in Machine Learning:
- Initialize neural network weights
- Split data into train/test sets
- Data augmentation
- Dropout regularization
- Monte Carlo simulations
"""

 # RANDOM NUMBER BASICS 

print("="*60)
print("RANDOM NUMBER GENERATION")
print("="*60)

#Set seed for reproducibility (Important!)
np.random.seed(42)

#Uniform distribution
uniform = np.random.random(5) # values beween 0 and 1. as float numbers
print("Uniform [0,1]:", uniform) 
print()

#Uniform distribution in range [low, high]
uniform_range = np.random.uniform(10, 20, size=5)
print("Uniform [10, 20]: ", uniform_range)
print()

# Random integers
random_ints = np.random.randint(1, 100, size=10)
print('Random integers [1, 100]: ', random_ints)
print()

 # NORMAL (GAUSSIAN) DISTRIBUTION

print("="*60)
print("NORMAL DISTRIBUTION (Most Common in ML!)")
print("="*60)

# Standard normal: mean=0, std=1
standard_norm = np.random.randn(5)
print("Standard normal (μ=0, σ=1): ", standard_norm)
print()

# Normal with custom mean and std
# Formula: mean + std*randn
mean, std = 100, 15
custom_norm = mean + std * np.random.randn(10)
print(f"Custom normal (μ={mean}, σ={std}):", custom_norm)
print()

# Using np.random.normal directly
normal = np.random.normal(loc=100, scale=15, size=10) #loc-Mean(Avg): peak of the distribution, scale-std: A measure of how spread out of numbers, size-10 elements of 1D array
print("Using np.random.normal: ", normal)
print()

# Verify distribution with large sample
large_sample = np.random.normal(100, 15, size=10000)
print(f"largest sample stats: ")
print(f"  Mean: {np.mean(large_sample):.2f} (should be ~100)")
print(f"   Std: {np.std(large_sample):.2f} (should be ~15)")
print()

 # ML APPLICATIONS
print("="*60)
print("ML APPLICATION: Weight Initialization")
print("="*60)

# Xavier/Glorot initialization (common for neural networks)
# Formula: randn @ sqrt(2 / (n_in + n_out))
n_inputs = 784 #MNST images (28x28)
n_outputs = 128 #Hidden layer neurons

#Initialization weights
weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / (n_inputs + n_outputs))
print(f"Weights Shape: {weights.shape}")
print(f"Weights mean: {np.mean(weights):.6f} (should be ~0)")
print(f"Weights std:   {np.std(weights):.6f}")
print()

 # RANDOM SAMPLING 

print("="*60)
print("RANDOM SAMPLING")
print("="*60)

# Random choice from array
options = np.array(['a','b','c','d','e'])
choices = np.random.choice(options, size=10)
print("Random Choices: ", choices)
print()

# With probabilities
probabilities = [0.1, 0.1, 0.3, 0.3, 0.2] #Must sum to 1
weighted_choices = np.random.choice(options, size=10, p=probabilities)
print("Weighted choices: ", weighted_choices)
print()

# Random permutation (shuffle)
arr = np.arange(10)
shuffled = np.random.permutation(arr)
print("Original: ", arr)
print("Shuffled: ", shuffled)
print()

 # TRAIN/TEST SPLIT EXAMPLE

print("="*60)
print("ML APPLICATION: Train/Test Split")
print("="*60)

#Simulate dataset: 1000 samples
n_samples = 1000
indices = np.arange(n_samples)

#Shuffle indicies
np.random.shuffle(indices)

# 80/20 Split
split_point = int(0.8 * n_samples)
train_indices = indices[:split_point]
test_indices = indices[split_point:]

print(f"Total samples: {n_samples}")
print(f"Train samples: {len(train_indices)}")
print(f"Test samples: {len(test_indices)}")
print(f"First 10 train indices: {train_indices[:10]}")
print()

 # REPRODUCIBILITY

print("="*60)
print("REPRODUCIBILITY with np.random.seed()")
print("="*60)

# Run 1
np.random.seed(123)
random1 = np.random.randn(5)
print("Run 1: ", random1)

#Run 2 (different seed)
np.random.seed(456)
random2 = np.random.randn(5)
print("Run 2: ", random2)

#Run 3 (different seed)
np.random.seed(123)
random3 = np.random.randn(5)
print("Run 3: ", random3)
print("Run 1 == Run3?", np.array_equal(random1, random3))
print()

print("💡 ALWAYS set seed in ML experiments for reproducibility!")