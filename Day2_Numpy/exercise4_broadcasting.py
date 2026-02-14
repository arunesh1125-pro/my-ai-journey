import numpy as np

# Task 1: Add a row vector to matrix using broadcasting
matrix = np.random.randint(1, 10, size=(4, 5))
row_vector = np.array([1,2,3,4,5])

print("Matrix: \n", matrix)
print("Row Vector: ", row_vector)
result = matrix + row_vector
print("Result: \n", result)
print()

# Task 2: Multiply matrix by column vector
col_vector = np.array([[2], [3], [4], [5]])
print("Column vector: \n",col_vector)
res1 = matrix * col_vector
print("Result:\n ", res1)
print()

# Task 3 : Normalize 2D DATASET
# Create a random dataset: 10 sample , 5 features

dataset = np.random.randn(10, 5) * 10 + 50 #reason for *10 + 50, intially randn is normalised random values , where std was 1 and mean was in 0 bcz of normalized format. now, multiplying through 10 becomes, new std value 1(old std) becomes 10(new std), and after adding + 50 ,new mean value becomes 50
print("Original datset: \n", dataset[:5 ])
print()

# Normalize using broadcasting
mean = np.mean(dataset, axis=0)
std = np.mean(dataset, axis=0)
normalized = (dataset - mean)/std

print("Normalized dataset (first 5 rows):\n", normalized[:5])
print("Mean per feature: ", np.mean(normalized, axis=0))
print("std per feature: ", np.std(normalized, axis=0))
