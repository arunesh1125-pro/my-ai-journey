import numpy as np

 # BRODCATSING : Numpy's Super Power

 # Broadcasting in NumPy is a powerful mechanism that allows arithmetic operations to be performed on arrays of different shapes. 
 # Instead of requiring arrays to have identical dimensions for element-wise operations, NumPy automatically expands the smaller array to match the shape of the larger one without making unnecessary copies in memory

# Rules:

# Prepending Ones: If the arrays have a different number of dimensions, the shape of the smaller-dimensional array is prepended with ones on the left.
# Dimension Compatibility: Two dimensions are compatible if they are equal, or if one of them is 1.
# Expansion: If a dimension is 1, the smaller array is "stretched" to match the larger array's size along that dimension.
#  Error: If dimensions disagree and neither is 1, a ValueError is raised.

#Example 1 : Scalar broadcasting
arr = np.array([1,2,3,4,5])
print("Array: ", arr)
print("Array + 10: ", arr +10 )
print()

#Example 2 : 1D to 2D broadcasting
matrix = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
row_vector = np.array([10,20,30])

print("matrix: \n",matrix)
print("Row vector: ", row_vector)
print()

#Add row vector to each row of matrix
result = matrix + row_vector
print("Matrix + row_vector:\n", result)
print() 

# Example 3: Column Broadcasting

col_vector = np.array([[10],
                      [20],
                      [30]])
print("Column vector:\n",col_vector)
print()
result1 = matrix + col_vector
print("Matrix + col_vector: \n", result1)
print()

# Real ML Example: Normalizd dataset

#Imagine; 5 sample, 3 features each

data = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12],
    [13,14,15]
])

print("Original Data: \n", data )
print()

#calaculate mean of each column (feature)
mean = np.mean(data, axis=0)
print("Mean of each feature: ", mean)
print()

#Calculate std of each column (Feature)
std = np.std(data, axis=0)
print("Std of each Feature: ",std)
print()

# Normalize : (data-mean)/std
# Broadcasting automatically expands mean and std
normalized_data = (data-mean)/std
print("Normalized Data:\n", normalized_data)
print()

#Verify: mean should be ~0, std should be ~1
print("Mean after normalization: ", np.mean(normalized_data, axis=0))
print("std after normalization: ", np.std(normalized_data, axis=0))
