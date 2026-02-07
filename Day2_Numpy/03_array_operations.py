import numpy as np

 # BASIC ARITHMETIC OPERATIONS

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([10,20,30,40,50])

print("arr1: ", arr1)
print("arr2: ", arr2)
print()

#Element-wise operations
print("Addition: ", arr1 + arr2)
print("Subtraction: ", arr2 - arr1)
print("Multiplication: ", arr1 * arr2)
print("Division: ", arr2 / arr1)
print("Power of 2 : ", arr1 ** 2)
print()

# Scalar Operations
print("Add 10: ", arr1 + 10)
print("Multiply by 2: ", arr1 * 2)
print("Square: ", arr2 **2 )
print()

 # UNIVERSAL FUNCTION

arr = np.array([1,4,9,16,25])

print("Square root: ", np.sqrt(arr))
print("Exponential: ", np.exp([1, 2, 3]))
print("Logarithm: ", np.log([1, 10, 100]))
print()

# Trigonometric functions
angles = np.array([0, np.pi/2, np.pi])
print("Sin: ", np.sin(angles))
print("Cos: ", np.cos(angles))
print()

#Rounding 
decimals = np.array([1.23, 4.56, 7.89])
print("Round: ", np.round(decimals))
print("Floor: ", np.floor(decimals))
print("Ceil: ", np.ceil(decimals))
print()

 # AGGREGATION FUNCTION

arr3 = np.array([2,7,2,9,1,5])

print("Array: ", arr3)
print("Sum: ", np.sum(arr3))
print("Mean: ", np.mean(arr3))
print("Median: ", np.median(arr3))
print("Standard Deviation: ", np.std(arr3))
print("Variance: ", np.var(arr3))
print("Min: ", np.min(arr3))
print("Max: ", np.max(arr3))
print("Index of min: ", np.argmin(arr3))
print("Index of max: ", np.argmax(arr3))
print()

# 2D aggregations
matrix = np. array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print("Matrix:\n", matrix)
print()
print("Sum of all elements: ", np.sum(matrix))
print("Sum of each Column: ", np.sum(matrix, axis=0))
print("Sum of each row: ", np.sum(matrix, axis=1))
print()
print("Mean of each Column: ", np.mean(matrix, axis=0))
print("Max of each row: ", np.max(matrix, axis=1))
print()

 # MATRIX OPERATIONS (CRITICAL FOR ML)

 # Dot product (1D Array)

v1= np.array([1,2,3])
v2=np.array([4,5,6])
dot_product = np.dot(v1, v2)
print("Dot Product: ", dot_product)
print()

# MAtrix Multiplication
a = np.array([[1, 2],
              [3, 4]])
b= np.array([[5, 6],
             [7, 8]])

#Method 1: np.dot
result = np.dot(a, b)
print("Matrix Multiplication: \n", result)
print()

# Method 2: @ operator (Python 3.5+)
result1 = a@b
print("Matrix Multiplication @ :\n", result1)
print()

# Element-wise multiplication (NOT matrix multiplication)
element_wise = a*b
print("Element-wise multiplication: \n", element_wise)
print()

#Transpose
print("Transpose of A: \n", a.T)
print()

#Matrix inverse (if exists)
from numpy.linalg import inv
a_inv = inv(a)
print("Inverse of a:\n", a_inv)
print("a*a_inv (should be identity):\n", a@a_inv)