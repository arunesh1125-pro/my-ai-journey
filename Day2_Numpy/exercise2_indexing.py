import numpy as np

# Create a test matrix
matrix = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20],
    [25,26,27,28,29]
])

print("Original Matrix:\n", matrix)
print()

#Task 1: Extract third row
third_row = matrix[2, :]
print("Third row: ", third_row)
print()

#task 2: Extract the second column
second_col = matrix[:, 1]
print("Second Column: ", second_col)
print()

#task 3: Extraxt 2x2 sub-matrix fromm centre
centre = matrix[1:3, 1:3]
print("Center 2x2:\n", centre)
print()

#task 4: Extract all elements greater than 15
greater_15 = matrix[matrix>15]
print("Elemts > 15: ", greater_15)
print()

#Task 5: Extract all even numbers
evens = matrix[matrix%2==0]
print("Even numbers: ", evens)
print()

#Task 6: Replace all values greater than 20 with 20
matrix_copy = matrix.copy()
matrix_copy[matrix_copy > 20] = 20
print("Matrix with cap at 20:\n", matrix_copy)