import numpy as np

 # ADVANCED BOOLEAN INDEXING

print("="*60)
print("AVANCED BOOLEAN INDEXING")
print("="*60)

# Create sample data
data = np.random.randint(1, 100, size=(10, 5))
print("Sample Data (10x5):\n", data)
print()

# Multiple conditions with AND (&)
mask1 = (data > 20) & (data < 80)
print("Elemets b/w 20 and 80: ")
print(data[mask1])
print()

# Multiple conditions with OR (|)
mask2 = (data < 10) | (data > 90)
print("Elements < 10 OR > 90: ")
print(data[mask2])
print()

# NOT Operation (~)
mask3 = ~(data % 2 == 0) # NOT even = odd
print("Odd numbers: ")
print(data[mask3])
print()

 # WHERE FUNCTION

print("="*60)
print("np.where() - Conditional Selection")
print("="*60)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Original: ", arr)
print()

# Replace values: if > 5, replace with 100, else keeep original
result = np.where(arr > 5, 100, arr)
print("After np.where (>5 -> 100)", result)
print()

# Multi-dimensional
matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

# Replace negatives with 0, positives with 1, zero stay 0
result2 = np.where(matrix > 5, 1, 0) # The function call is structured as: np.where(condition, value_if_true, value_if_false)
print("Matrix:\n", matrix) # value_if_true (1), value_if-false (0)
print("Binary (>5 = 1):\n", result2)
print()

 # ARGWHERE - Find indices where condition is True

print("="*60)
print("np.argwhere - Find Indices")
print("="*60)

data = np.array([[10,25,30],
                 [15,50,35],
                 [20,45,40]])

# Find where values > 30
indices = np.argwhere(data>30)
print("Data:\n", data)
print("Indices where > 30:\n", indices)
print("Values at those indices: ", data[data>30])
print()

 # ANY and ALL

print("="*60)
print("np.any() and np.all()")
print("="*60)

arr1 = np.array([1,3,5,7,9])
print("Array: ", arr1)
print("Any elements >5?: ", np.any(arr1>5))
print("All elements > 5?: ", np.all(arr1 > 5))
print("All elements odd?: ", np.all(arr % 2 == 1))
print()

# With axis parameter
matrix1 = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print("Matrix:\n", matrix1)
print("Any column has all values > 5?: ")
print(np.all(matrix > 5, axis=0))
print("Any row has all values > 5?: ")
print(np.all(matrix1>5, axis=1))
print()

 # UNIQUE, BINCOUNT, HISTOGRAM

print("="*60)
print("UNIQUE VALUES and COUNTS")
print("="*60)

data1 = np.array([1,2,2,3,3,3,4,4,4,4])
unique_vals = np.unique(data1)
print("Data: ", data)
print("Unique values: ", unique_vals)
print()

# Get counts too
unique_vals, counts = np.unique(data1, return_counts = True)
print("\nValue: Count")
for val, count in zip(unique_vals, counts):
    print(f"   {val}  :  {count}   ")
print()

# Bincount (for non-negative integers)
bincount = np.bincount(data1)
print("Bincount: ", bincount) #np.bincount finds the occurence(frequency of each value in the sample, starts from integer 0 to N, so, that's why the output [0 1 2 3 4], because integer 0 - 0(frequency), 1- 1 (frequency, 2- 2(frquency)))
print("(Index is value, value is count)")
print()

# Histogram
data2 = np.random.randn(1000)*10+50 # Std=10, Mean=50, why because, after we used the random.randn, it is normalised one (called as Normal (Guassian) Distribution), make the std=1, mean=0, so,to make it as desired one, we *10 to std and +50 to mean
hist, bin_edges = np.histogram(data2, bins=5)
print("Histogram of 1000 random values: ")
print("Count per bin: ", hist)
print("Bin edges: ", bin_edges)
