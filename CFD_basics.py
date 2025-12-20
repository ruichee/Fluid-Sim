import numpy as np
import matplotlib.pyplot as plt


# uniform mesh

class Meshing:

    def __init__(self, x_size, y_size):
        self.x_lim = x_size
        self.y_lim = y_size

    def generate(self):
        self.mesh_grid = np.zeros((self.y_lim, self.x_lim))


class Boundary_Conditions:

    def __init__(self):
        pass


class LinAlg_direct:

    def __init__(self):
        pass

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


class LinAlg_iterative:

    def __init__(self):
        pass

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
        L_star = A - np.triu(A)     # lower triangle in A without diags, not the same as U in LU decomposition
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
print(LinAlg_direct.LU_decomp(A))

A1 = np.array([[8, 2, 0], [3, -5, 7], [-2, 1, 9]])
b1 = np.array([[12, 14, 27]]).T
u1_0 = np.array([[3, 2, 1]]).T
print(LinAlg_iterative.Jacobi(A1, b1, u1_0))
print(LinAlg_iterative.Gauss_Seidel(A1, b1, u1_0))
print(LinAlg_iterative.SOR(A1, b1, u1_0, 0.8))






x_size = 100
y_size = 100

u = np.zeros((y_size+2, x_size+2))

u[1:101, 0] = 1000

print(u)
#plt.pcolormesh(range(42), range(42), u)
#plt.show()

for i in range(300):
    for x in range(1, 101):
        for y in range(1, 101):
            u[[0, 101], :] = 0
            u[x,y] = (u[x-1, y] + u[x+1, y] + u[x, y-1] + u[x, y+1]) / 4

plt.pcolormesh(range(102), range(102), u, cmap="jet")
plt.show()

