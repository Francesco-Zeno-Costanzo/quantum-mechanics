"""
Code to solve the Schrodinger equation in 3 spatial dimensions via diagonalization
We consider only harmonic oscillator. With big values of N, i.e. N=100/150 the
diagonalization take 200 - 1500 seconds. So after the computation we save all
eigenvalues and eigenvectors to plot them with a second code.
"""
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import kron, diags, eye


def U(x, y, z):
    ''' Potenteial
    '''
    return 0.5 * (x**2 + y**2 + z**2)


def diag_H(N, k, dx):
    '''
    Function to build the hamiltonian and diagonalization

    Parameters
    ----------
    N : int
        number of points
    k : int
        numer of eigenvalues and eigenstates to found
    dx : float
        grid spacing
    '''
    #=========================================================
    # Build hamiltonian of system and diagonalization
    # By discretizing the 3 spatial derivatives, since they
    # are different coordinates, the tensor product between the
    # matrices and the identity must be made:
    # We do it first between x and y and then we do the same
    # with what we have obtained and z thanks to kron function
    #=========================================================

    # Create the 3D Hamiltonian matrix
    Px, Py, Pz = 3*[diags([1, -2, 1], [-1, 0, 1], shape=(N, N))]
    Ix, Iy, Iz = 3*[eye(N)]

    # Combine the 3 individual kinetical term using Kronecker products
    Pxy = kron(Iy, Px) + kron(Py, Ix)
    Ixy = kron(Iy, Ix)
    P   = kron(Iz, Pxy) + kron(Pz, Ixy)
    V   = diags(U(X, Y, Z).reshape(N**3), 0)
    H   = -1/(2*dx**2)*P + V

    eigval, eigvec = eigsh(H, k=k, which='SM')

    return eigval, eigvec

#=========================================================
# Computational parameters
#=========================================================

N       = 150                       # number of points
x       = np.linspace(-7, 7, N)     # grid
X, Y, Z = np.meshgrid(x, x, x)      # 3D grid
dx      = np.diff(x)[0]             # size step
k       = 10                        # number of eigvalues

start = time.time()
eigval, eigvec = diag_H(N, k, dx)
print(f"{time.time()-start} s")

# Save data
np.savez(f"data_{N}", eigval, eigvec)

#=========================================================
# Quickly comparison to se if all is ok
#=========================================================

deg = lambda n: int(0.5*(n + 1)*(n + 2))

E_n = [(n + 3/2) for n in range(4) for g in range(deg(n))]

for i in range(len(eigval)):
    print(f"{eigval[i]:.5f}, {E_n[i]}")
