import numpy as np
 
 #CREATING ARRAYS

#From Python List
list_1d = [1,2,3,4,5]
array_1d = np.array(list_1d)
print("1D Array: ", array_1d)
print("Type: ", type(array_1d))
print("DataType: ", array_1d.dtype)
print()

# 2D Array (Matrix)
list_2d = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]
array_2d = np.array(list_2d)
print("2D Array:\n", array_2d)
print("Shape:", array_2d.shape)
print("Dimensions: ", array_2d.ndim)
print("Size: ", array_2d.size) # 9 total elements
print()

# 3D Array (Matrix)
array_3d = np.array([[[1,2], [3,4]],
                      [[5,6],[7,8]]])
print("3D Array:\n", array_3d)
print("Shape: ",array_3d.shape)
print()

 #ARRAY CREATION FUNCTIONS

# Zeros (Useful for initialization)
zeros = np.zeros((3, 4)) # 3 rows, 4 columns of zeros
print("Zeros: ", zeros)
print()

#Ones
ones = np.ones((2, 3)) #2 rows, 3 columns
print("Ones: ", ones)
print()

#Full (fill with specific value)
fives = np.full((2,2),5) #2 rows, 2 columns with 5's
print('Array of 5s: \n', fives)
print()

#Identity matrix (diagonal 1s)
identity = np.eye(4) #eye- 2d matrix with identity 1's on 4x4 matrix array
print("Identity Matrix:\n", identity)
print()

#Range of values
range_array = np.arange(0, 10, 2) # Start, Stop, Step
print("Range: ", range_array)
print()

#Evenly Spaced values
linspace = np.linspace( 0, 1, 5) #Start, Stop, number if Values
print("Linspace: ", linspace)
print()

# Random arrays (Important for ML)
random_array = np.random.random((3, 3)) # Values between 0 and 1 - 3 rows and 3 columns
print("Random: ", random_array)
print()

random_int = np.random.randint(0, 100, size=(3, 3)) # Rand Values b/w 0 and 100 with 3x3 matrix
print("Random Integers:\n", random_int)
print()

# Normal Distribution
normal = np.random.randn(5) #Mean=0, Std=1
print('Normal Distribution: ', normal)
print()

 # DATA TYPES

# Specific data type
float_array = np.array([1, 2, 3], dtype=np.float32)
print("Float array: ", float_array)
print("Data Type: ", float_array.dtype)
print()

#Convert data type
int_to_float = array_1d.astype(np.float64) # .astype(np.int32)
print("Converted to float: ", int_to_float)
print("Data Type: ", int_to_float.dtype)
print()

#