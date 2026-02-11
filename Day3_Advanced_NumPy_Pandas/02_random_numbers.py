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

# Normal ith custom mean and std
# Formula: mean + std*randn
mean, std = 100, 15
custom_norm = mean + std * np.random.randn(10)
print(f"Custom normal (μ={mean}, σ={std}):", custom_norm)
print()

# Using np.random.normal directly
normal = np.random.normal(loc=100, scale=15, size=10) #loc-Mean(Avg): peak of the distribution, scale-std: A measure of how spread out of numbers, size-10 elements of 1D array
print("Using np.random.normal: ", normal)
print()