"""
MATRIX CALCULATOR
A command-line tool for matrix operations
This is our 3rd portfolio project!
"""

import numpy as np

def print_menu():
    print("\n"+"="*50)
    print("MATRIX CALCULATOR")
    print("="*50)
    print("1. Create Matrix")
    print("2. Add Matrices")
    print("3. Subtract Matrices")
    print("4. Multiply Matrices")
    print("5. Transpose Matrix")
    print("6. Matrix Inverse")
    print("7. Matrix Determinant")
    print("8. Element-wise Operations")
    print("9. Statistics")
    print("10. Exit")
    print("="*50)

def create_matrix():
    """Create a matrix from user input"""
    try:
        rows = int(input("Enter number of row: "))
        cols = int(input("enter number of columns: "))

        print(f"\nEnter {rows}x{cols} matrix elements (space-separated rows):")
        matrix = []
        for i in range(rows):
            row = list(map(float, input(f"Row {i+1}: ").split()))
            if len(row) != cols:
                print(f"Error: Expected {cols} elements, got {len(row)}")
                return None
            matrix.append(row)
            
        return np.array(matrix)
    except ValueError:
        print("Invalid Input!")
        return None

def display_matrix(matrix, name="Matrix"):
    """Pretty print a matrix"""
    print(f"\n{name}:")
    print(matrix)
    print(f"Shape: {matrix.shape}")

def add_matrices():
    """Add two matrices"""
    print("\n---ADD MATRICES---")
    print("Matrix A: ")
    A = create_matrix()
    if A is None:
        return
    print("\nMatrix B: ")
    B = create_matrix()
    if B is None:
        return
    
    if A.shape != B.shape:
        print(f"Error: Shape don't match! {A.shape} vs {B.shape}")
        return
    
    result = A+B
    display_matrix(A, "Matrix A")
    display_matrix(B, "Matrix B")
    display_matrix(result, "A+B")

def subtract_matrices():
    """Subtract two matrices"""
    print("\n---SUBTRACT MATRICES---")
    print("MatrixA: ")
    A = create_matrix()
    if A is None:
        return
    
    print("\nMatrix B: ")
    B = create_matrix()
    if B is None:
        return
    
    if A.shape != B.shape:
        print(f"Error: Shapes don't match! {A.shape} vs {B.shape}")
        return
    
    result = A - B
    display_matrix(A, "Matrix A")
    display_matrix(B, "Matrix B")
    display_matrix(result, "A - B")

def multiply_matrices():
    """Multiply two matrices"""
    print("\n--- MULTIPLY MATRICES ---")
    print("Matrix A:")
    A = create_matrix()
    if A is None:
        return
    
    print("\nMatrix B:")
    B = create_matrix()
    if B is None:
        return
    
    if A.shape[1] != B.shape[0]:
        print(f"Error: Cannot multiply! A cols ({A.shape[1]}) must equal B rows ({B.shape[0]})")
        return
    
    result = A @ B
    display_matrix(A, "Matrix A")
    display_matrix(B, "Matrix B")
    display_matrix(result, "A × B")

def transpose_matrix():
    """Transpose a matrix"""
    print("\n--- TRANSPOSE MATRIX ---")
    A = create_matrix()
    if A is None:
        return
    
    result = A.T
    display_matrix(A, "Original Matrix")
    display_matrix(result, "Transposed Matrix")

def matrix_inverse():
    """Calculate matrix inverse"""
    print("\n--- MATRIX INVERSE ---")
    A = create_matrix()
    if A is None:
        return
    
    if A.shape[0] != A.shape[1]:
        print("Error: Matrix must be square")
        return
    
    try:
        result = np.linalg.inv(A)
        display_matrix(A, "Original Matrix")
        display_matrix(result, "Inverse Matrix")

        # Verify A x A^(-1) = I
        identity = A @ result
        display_matrix(identity, "A x A^(-1) (should be identity)")
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular (non-invertible)")
    
def matrix_determinant():
    """Calculate matrix determinant"""
    print("\n--- MATRIX DETERMINANT ---")
    A = create_matrix()
    if A is None:
        return
    
    if A.shape[0] != A.shape[1]:
        print("Error: Matrix must be square!")
        return
    
    det = np.linalg.det(A)
    display_matrix(A, "Matrix")
    print(f"\nDeterminant: {det:.4f}")

def element_wise_operations():
    """Perform element-wise operations"""
    print("\n--- ELEMENT-WISE OPERATIONS ---")
    A = create_matrix()
    if A is None:
        return
    
    display_matrix(A, "Original Matrix")

    print("\nSelect operations: ")
    print("1. Square each element")
    print("2. Square root")
    print("3. Exponential")
    print("4. Logarithm")
    print("5. Add scalar")
    print("6. Multiply by scalar")

    choice = input("Choice: ")

    if choice == "1":
        result = A ** 2
        display_matrix(result, "Squared")
    elif choice == "2":
        if np.any(A < 0):
            print("Error: Cannot take square root of negative numbers")
            return
        result = np.sqrt(A)
        display_matrix(result, "Square Root")

    elif choice == "3":
        result = np.exp(A)
        display_matrix(result, "Exponential")
    elif choice == "4":
        if np.any(A<=0):
            print("Error: Logarithm requires positive numbers")
            return
        result = np.log(A)
        display_matrix(result, "Natural Log")
    elif choice == "5":
        scalar = float(input("Enter scalar to add: "))
        result = A + scalar
        display_matrix(result, f"Matrix + {scalar}")
    elif choice == "6":
        scalar = float(input("Enter a scalar to multiply: "))
        result = A * scalar
        display_matrix(result, f"Matrix x {scalar}")
    else:
        print("Invalid choice!")

def matrix_statistics():
    """Calculate matrix statistics"""
    print("\n--- MATRIX STATISTICS ---")
    A = create_matrix()
    if A is None:
        return
    
    display_matrix(A, "Matrix")

    print(f"\nStatistics: ")
    print(f"Sum: {np.sum(A):.4f}")
    print(f"Mean: {np.mean(A):.4f}")
    print(f"Median: {np.median(A):.4f}")
    print(f"Std Dev: {np.std(A):.4f}")
    print(f"variance: {np.var(A):.4f}")
    print(f"Min: {np.min(A):.4f}")
    print(f"max: {np.max(A):.4f}")
    print(f"\nRow sums: {np.sum(A, axis=1)}")
    print(f"Column sums: {np.sum(A, axis=0)}")

def main():
    """Main program loop"""
    print("Welcome to Matrix Calculator!")

    while True:
        print_menu()
        choice = input("\nEnter your choice (1-10): ")

        if choice == "1":
            matrix = create_matrix()
            if matrix is not None:
                display_matrix(matrix)
        elif choice == "2":
            add_matrices()
        elif choice =="3":
            subtract_matrices()
        elif choice == "4":
            multiply_matrices()
        elif choice == "5":
            transpose_matrix()
        elif choice == "6":
            matrix_inverse()
        elif choice == "7":
            matrix_determinant()
        elif choice == "8":
            element_wise_operations()
        elif choice == "9":
            matrix_statistics()
        elif choice == "10":
            print("\nThank you for using Matrix Calculator!")
            print("Made with NumPy💪")
            break
        else:
            print("Invalid choice! Please try again.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()