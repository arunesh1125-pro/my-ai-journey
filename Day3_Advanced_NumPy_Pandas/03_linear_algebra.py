import numpy as np
from numpy.linalg import inv, det, eig, solve, norm

"""
Linear Algebra is the LANGUAGE of Machine Learning!
Every ML algorithm uses these operations.
"""

 # MATRIX OPERATIONS REVIEW

print("="*60)
print("MATRIX OPERATIONS")
print("="*60)

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

print("Matrix A:\n", a)
print("Matrix B:\n", b)
print()

# Matrix multiplication
c = a@b
print("A @ B:\n", c)
print()

# Transpose
print("A Transpose:\n", a.T)
print()

 # DETERMINANT

print("="*60)
print("DETERMINANT")
print("="*60)

det_a = det(a)
print("det(a):", det_a)
print()

# Determinant tells if matrix is invertible
# det ≠ 0 -> invertible
# det = 0 -> singular (not invertible)

singular = np.array([[1, 2],
                     [2, 4]])   # Second row = 2 x first row
print("Singular matrix:\n", singular)
print("det(singular): ", det(singular))
print()

 # MATRIX INVERSE

print("="*60)
print("MATRIX INVERSE")
print("="*60)

A_inv = inv(a)
print("A inverse:\n", A_inv)
print()

# Verify: A@ A^(-1) = I
identity = a @ A_inv
print("A @ A^(-1) (should be identity):\n", identity)
print()

 # SOLVING LINEAR SYSTEMS

print("="*60)
print("SOLVING LINEAR EQUATIONS: Ax = b")
print("="*60)

# System:
# 2x + 3y = 8
# 5x + 4y = 13

A = np.array([[2,3],
              [5,4]])
B = np.array([8, 13])

# Solve for x

x = solve(A, B)
print("Coefficients matrix A:\n", A)
print("Constants vector b:", B)
print("\nSolution x: ", x)

# Verify
result = A @ x
print("Verification A @ x: ", result)
print("Should be equal B: ", B)
print()

 # EIGEN VALUES and EIGEN VECTORS

print("="*60)
print("EIGEN VALUES and EIGEN VECTORS")
print("="*60)

A1 = np.array([[4,2],
               [1,3]])

eigenvalues, eigenvectors = eig(A1)

print("Matrix A1:\n", A1)
print("\nEigen values: ", eigenvalues)
print("\nEigen vectors:\n", eigenvectors)
print()

# Verify: A1 @ v = λ @ v
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]

    left = A1 @ v
    right = λ * v

    print(f"Eigenvalue {i+1}: {λ:.4f}")
    print(f"  A @ v = {left}")
    print(f"  λ * v = {right}")
    print(f"  Equal? {np.allclose(left, right)}")
    print()

 # NORMS (Vector/Matrix Magnitude)

print("="*60)
print("VECTOR and MATRIX NORMS")
print("="*60)

v = np.array([3, 4])
print("Vector v: ", v)
print()

# L2 norm (Euclidean distance)
l2 = norm(v)
print(f"L2 norm (length): {l2}")  # sqrt(3^2 + 4^2) = 5
print()

# L1 norm (Manhattan distance)
l1 = norm(v, ord=1)
print(f"l1 norm: {l1}")  # |3| + |4| = 7
print()

# Infinity norm (max absolute value)
linf = norm(v, ord=np.inf)
print(f"L-infinity norm: {linf}")   # max(3, 4) = 4
print()

# Matrix norms
A = np.array([[1,2],
              [3,4]])
print('Matrix A:\n', A)
print(f"Frobenius norm: {norm(A, 'fro')}")  #sqrt(sum of squared elements)
print()

 # ML APPLICATION : Linear Regression (Closed Form) - (OLS Method: Ordinary Least Square Method)

print("="*60)
print("ML APPLICATIONS: Linear Regression")
print("="*60) 

# Generate synthetic data: y = 3x + 2 + noise
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y_true = 3 * X.squeeze() + 2
y = y_true + np.random.randn(50) * 2  # Add noise

#Add bias term (column of ones)
x_with_bias = np.c_[np.ones(len(X)), X]  # np.c_[]: This is a NumPy function that performs column-wise concatenation. It takes the newly created column of ones and appends it to the left side of the existing data matrix X. 

#Closed-form solution: 0 = (X^T X)^(-1) X^T y
theta = inv(x_with_bias.T @ x_with_bias) @ x_with_bias.T @ y   #  (theta) represents the vector of the learned parameters (intercept and slope).X is the X_with_bias matrix.yis the noisy y-data.

print("True parameters: slope=3, intercept=2")
print(f"Learned parameters: slope={theta[1]:.2f}, intercept={theta[0]:.2f}")
print()

# Predictions
y_pred = x_with_bias @ theta

# Calculate error (mse)

mse = np.mean((y - y_pred)**2)
print(f"Mean Squared Error: {mse:.4f}")