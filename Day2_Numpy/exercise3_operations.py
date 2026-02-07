import numpy as np

#Task 1: Create two arrays and perform all arithmetic operations
a = np.array([10,20,30,40,50])
b = np.array([5, 10, 15, 20, 25])

print("Array a: ", a)
print("Array b: ", b)
print("a+b: ", a+b)
print("a-b: ", a-b)
print("a*b: ", a*b)
print("a/b: ", a/b)
print("a**2: ", a**2)
print()

#Task 2: Calculate statistics
data = np.random.randint(1, 100, size=50)
print("Random data (first 100): ", data[:10])
print("Mean: ", np.mean(data))
print("Median: ", np.median(data))
print("Std Dev: ", np.std(data))
print("Min: ", np.min(data))
print("Max: ", np.max(data))
print()

#Task 3: Matrix Operations
x = np.array([[1,2,3],
              [4,5,6]])
y = np.array([[7,8],
              [9,10],
              [11,12]])

#Matrix multiplication
result = x @ y
print("X Shape: ", x.shape)
print("y shape: ", y.shape)
print("x@y: ", result)
print("Result Shape: ", result.shape)
print()

# Task 4: Normalize an array (import for Ml)
#Formula: (x-mean)/std
arr = np.array([10,20,30,40,50])
mean = np.mean(arr)
std = np.std(arr)
normalized = (arr - mean) / std
print("Original: ", arr)
print("Normalized: ", normalized)
print("Normalized mean: ", np.mean(normalized))
print("Normalized std: ", np.std(normalized))