import numpy as np
from numpy.linalg import inv, det, eig, norm

# Task 1: Check if matrices are invertible
matrices = [
    np.array([[2,3],[1,4]]),
    np.array([[1,2],[2,4]]), #Singular
    np.array([[5,6],[7,8]])
]

for i, M in enumerate(matrices):
    print(f"Matrix {i+1}:\n{M}")
    d = det(M)
    print(f"Determinant: {d:.4f}")

    if abs(d) > 1e-10:      #Not zero (accounting for floating point)
        print("✅ Invertible")
        M_inv = inv(M)
        print(f"Inverse:\n{M_inv}")
    else:
        print("❌ Singular (not invertible)")
    print()

# Task 2 : Solve system of equations
# 3x + 2y + z = 10
# 2x + 3y + 2z = 14
# x + 2y + 3z = 14

A = np.array([[3,2,1],
              [2,3,2],
              [1,2,3]])
b = np.array([10, 14, 14])
x = np.linalg.solve(A, b)
print("system of solution:", x)
print("verification A @ x: ", A @ x)
print("should equal b: ", b)
print()

#Task 3: Principal Component Analysis (PCA) simplified
# Generate 2D data
np.random.seed(42)
data = np.random.randn(100, 2)
#Add correlation
data[:, 1] = data[:, 0] * 0.8 + np.random.randn(100) * 0.3 # Second feature = first feature * 0.8 -> correlated + adding noise of *0.3

print("Data Shape: ", data.shape)
print("Data mean: ", np.mean(data, axis=0))
print()

# Centre the data
data_centered = data - np.mean(data, axis=0)

# Calculate covariance matrix
cov_matrix = np.cov(data_centered.T)
print("Covariance matrix:\n", cov_matrix)
print()

# Find Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = eig(cov_matrix)
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print()

# Principle component = eigenvector with largest eigenvalue
pc1_idx = np.argmax(eigenvalues)
principle_component = eigenvectors[:, pc1_idx]
print("Principle component (PC1): ", principle_component)
print("Explained variance: ", eigenvalues[pc1_idx]/np.sum(eigenvalues))