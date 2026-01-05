import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from linear_solvers import Iterative as iter

#########################################################################################################

## Lid-Driven Cavity Flow (streamfunction-vorticity formulation) ##

## poisson equation will be solved via Jacobi (Au=b) rather than stencil ##

#########################################################################################################

def poisson_assemble(nx, ny):
    A = np.zeros((nx-1)*(ny-1))
    # to be implemented

    return A


def poisson_solve(A, b, n_iter, tol):
    stmfunc0 = np.zeros(np.size(b))
    stmfunc = iter.Jacobi(A, b, stmfunc0)
    return stmfunc


def main(Lx, Ly, dt, nx, ny, nt, vort, Re, n_iter, tol):

    dx = Lx/nx
    dy = Ly/ny
    
    A = poisson_assemble(nx, ny)
    b = np.array() # to be derived
    vort = np.zeros((nx-1)*(ny-1))

    for _ in range(nt):
        stmfunc = poisson_solve(A, b, n_iter, tol)
        vort_0 = vort.copy()

        # explicit forward euler
        vort = vort_0 # to be derived
    


def get_uv():
    pass 
# use staggered grid


def plot():
    pass

