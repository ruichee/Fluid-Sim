import numpy as np
import matplotlib.pyplot as plt


class Direct:

    def LU_decomp(A: np.ndarray) -> tuple[np.ndarray, np.ndarray] :
        # assumes no row swaps necessary, matrix is such that rows with earlier entries are always higher
        # also assumes no zero in the main diagonal
        # try to implement a way to rearrange the matrix if row swaps are necessary
        A_size = A.shape[0]
        L_inv_matrix = np.identity(A_size)
        A_new_matrix = A

        for j in range(A_size-1):
            L_j_matrix = np.identity(A_size)

            for i in range(j+1, A_size):
                L_j_matrix[i, j] = -A_new_matrix[i, j] / A_new_matrix[j, j]
            
            L_inv_matrix = L_j_matrix @ L_inv_matrix
            A_new_matrix = L_j_matrix @ A_new_matrix

        U_matrix = A_new_matrix
        L_matrix = np.linalg.inv(L_inv_matrix)
    
        return L_matrix, U_matrix

    def QR_decomp():
        pass


class Iterative:

    def residual_error(norm_type: str = "Euclidean"):
        
        match norm_type:

            case "Euclidean":
                pass

            case "Maximum":
                pass

            case "RMS":
                pass

        

    def Jacobi(A: np.ndarray, b: np.ndarray, u_0: np.ndarray):
        
        D = np.diag(np.diag(A))
        D_inv = np.linalg.inv(D)
        R = A - D

        u_old = u_0
        k = 0
        residual_array = []
        
        while True:
            u_new = -D_inv @ R @ u_old + D_inv @ b
            
            residual = np.linalg.norm(u_new - u_old)
            residual_array.append(residual)
            if residual < 1e-12:
                break
            k += 1
            u_old = u_new
        
        plt.semilogy(range(k+1), residual_array)
        plt.show()
        return u_new


    def Gauss_Seidel(A: np.ndarray, b: np.ndarray, u_0: np.ndarray):
        
        D = np.diag(np.diag(A))
        L = np.tril(A)      # lower triangular matrix part of A, not the same as L in LU decomposition
        U_star = A - L      # upper triangle in A without diags, not the same as U in LU decomposition
        L_inv = np.linalg.inv(L)

        u_old = u_0
        k = 0
        residual_array = []

        while True:
            u_new = -L_inv @ U_star @ u_old + L_inv @ b

            residual = np.linalg.norm(u_new - u_old)
            residual_array.append(residual)
            if residual < 1e-12:
                break
            k += 1
            u_old = u_new

        plt.semilogy(range(k+1), residual_array)
        plt.show()
        return u_new


    def SOR(A: np.ndarray, b: np.ndarray, u_0: np.ndarray, relaxation: float):
        
        w = relaxation
        D = np.diag(np.diag(A))
        L_star = A - np.triu(A)     # lower triangle in A without diags, not the same as L in LU decomposition
        U_star = A - np.tril(A)     # upper triangle in A without diags, not the same as U in LU decomposition
        DwL_inv = np.linalg.inv(D + w*L_star)

        u_old = u_0
        k = 0
        residual_array = []

        while True:
            u_new = DwL_inv @ (w*b - (w*U_star + (w-1)*D) @ u_old) 

            residual = np.linalg.norm(u_new - u_old)
            residual_array.append(residual)
            if residual < 1e-12:
                break
            k += 1
            u_old = u_new

        plt.semilogy(range(k+1), residual_array)
        plt.show()
        return u_new           
        

A = np.array([[1, 2, 1], [-1, -1, 3], [-2, 1, 1]])
# print(Direct.LU_decomp(A))

A1 = np.array([[8, 2, 0], [3, -5, 7], [-2, 1, 9]])
b1 = np.array([[12, 14, 27]]).T
u1_0 = np.array([[3, 2, 1]]).T
# print(Iterative.Jacobi(A1, b1, u1_0))
# print(Iterative.Gauss_Seidel(A1, b1, u1_0))
# print(Iterative.SOR(A1, b1, u1_0, 0.8))





############################## HEAT TRANSFER EXAMPLE ##############################

# Steady State Heat Diffusion in Rectangular Plate
# where one edge is heated, while other 3 edges are maintained at low temperature

# Governing Equation: ∂²T/∂x² + ∂²T/∂y² = ∇²T = 0
# Boundary Condition: T(x=0) = 100, T(y=0) = T(y=100) = T(x=100) = 0

# Discretized: T(i,j) = ( T(i-1,j-1) + T(i+1,j-1) + T(i-1,j+1) + T(i+1,j+1) ) / 4
# => average of 4 neighbouring points

