import numpy as np

 # RESHAPING

arr = np.arange(12)
print("Original: ", arr)
print("Shape: ", arr.shape)
print()

#Reshape to 2D
reshaped_2d = arr.reshape(3, 4)
print("Reshaped to 3x4:\n", reshaped_2d)
print()

#Reshape to 3d
reshaped_3d = arr.reshape(2, 3, 2)
print("Reshaped to 2x3x2: \n", reshaped_3d)
print()

# Flatten back to 1D
flattened = reshaped_2d.flatten()
flatten3d = reshaped_3d.flatten()
print("Flattened 2D: ", flattened)
print("Flattened 3D: ", flatten3d)
print()

# Ravel (like flatten, but returns view when possible)
raveled = reshaped_2d.ravel()
print("Raveled: ", raveled)
print()

 # STACKING

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack (row-wise)
v_stack = np.vstack((a,b))
print("Vertical Stack:\n", v_stack)
print()

#Horizontal stack (Column-wise)
h_stack = np.hstack((a,b))
print("Horizontal Stack: ", h_stack)
print()

 # SPLITTING

arr1 = np.arange(12)
split = np.split(arr1, 3)
print("Split into 3:", split)
print()

matrix = np.arange(12).reshape(4,3)
print("Matrix: \n", matrix)
print()

#Split horizontally (row-wise)
hsplit = np.hsplit(matrix, 3) #3 columns -> 3 arrays
print("Horizontal Split: ")
for i, part in enumerate(hsplit):
    print(f"Part {i}: \n", part)
print()

# Split Vertically (column-wise)
vsplit = np.vsplit(matrix, 2) # 4 rows -> 2 arrays of rows each
print("Vertical Split: ")
for i, part in enumerate(vsplit):
    print(f"Part {i}:\n", part)
print()