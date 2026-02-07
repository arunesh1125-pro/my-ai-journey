import numpy as np

# 1D ARRAY INDEXING

arr = np.array([10,20,30,40,50])
print("First element: ", arr[0])
print("Last element: ", arr[-1])
print("Third element: ", arr[2])
print()

# Slicing
print("FIrst 3 elements: ", arr[0:3])
print("From index 2 onwards: ", arr[2:])
print("Last 3 elements: ", arr[-3:])
print("Every second element: ", arr[::2])
print("Reverse array: ", arr[::-1])
print()

 # 2D ARRAY INDEXING

matrix = np.array([
    [1,2,3,4,],
    [5,6,7,8],
    [9,10,11,12]
])

print("matrix:\n", matrix)
print()

# Access specific element [row, column]
print("Element at [0, 0]: ", matrix[0, 0])
print("Element at [1, 2]: ", matrix[1, 2])
print("Element at [2, 3]: ", matrix[2, 3])
print()

#Access entire row
print("First row: ", matrix[0, :])
print("second Row: ", matrix[1, :])
print()

#Access entire column
print("First column: ", matrix[:, 0])
print("Third COlumn: ", matrix[:, 2])
print()

#Slicing 2D arrays
print("First 2 rows, first 2 colums:\n", matrix[0:2, 0:2])
print()

print("All rows, column 1 to 3:\n", matrix[:, 1:3])
print()

# BOOLEAN INDEXING (Useful for ML)

arr1 = np.array([1,2,3,4,5,6,7,8,9,10])

#Create boolean mask
mask = arr1 > 5
print("Mask (arr1>5):", mask)
print()

#Filter using mask
filtered = arr1[mask]
print('Elements > 5:', filtered)
print()

# Direct filtering (one Line)
print("Elements divisible by 2: ", arr1[arr1%2==0])
print("Elements between 3 and 7: ", arr1[(arr1>=3) & (arr1<=7)])
print()

 # FANCY INDEXING

arr2 = np.array([10,20,30,40,50,60])

# Select specific indices
indicies = [0,2,4]
selected = arr2[indicies]
print("Elements at indices [0, 2, 4]: ", selected)
print()

# 2D fancy indexing
matrix1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

rows = [0,2]
columns = [1,2]
print("Selected elements: ", matrix1[rows,columns]) #matrix1[rows(1,1), column(2,2)] it get transposed to look and coresponds to row(1,2), column(1,2)

