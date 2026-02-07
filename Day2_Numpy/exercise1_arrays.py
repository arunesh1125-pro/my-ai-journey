import numpy as np

# Task 1: Create a 5x5 matrix of all zeros
zeros_5x5 = np.zeros((5,5))
print("5x5 Zeros:\n",zeros_5x5)
print()

#Task 2: Create a 4x4 Identical matrix
identity_4x4 = np.eye(4)
print("4x4 Identity:\n", identity_4x4)
print()

#Task 3: Create an array from 10 to 50 with step of 5
array_1 = np.arange(10, 50, 5)
print("Array of 10 to 50 (step 5):", array_1)
print()

# Taskk 4: Create 10 evenly space numbers between 0 and 5
array_2 = np.linspace(0, 5, 10)
print("Evenly Spaced: ", array_2)
print()

#Task 5: Create a 3x3 matrix with random values b/w 0 and 1
random_3x3 = np.random.random((3, 3))
print("Random 3x3:\n", random_3x3)
print()

# Task 6: Create a 10-element array with value from 1 to 100,
# then reshape it to 2x5
array_1_100 = np.arange(1, 11)
reshaped = array_1_100.reshape(2, 5)
print("Reshaped to 2x5:\n", reshaped)
print()