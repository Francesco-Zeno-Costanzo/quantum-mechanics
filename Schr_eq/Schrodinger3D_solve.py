"""
Code to solve the Schrodinger equation in 3 spatial dimensions via diagonalization
We consider harmonic oscillator and Hydrogen atom. With big values of N, i.e. N=100/150
the diagonalization take a lot of time (i.e. 25-100 minutes). So after the computation
we save all eigenvalues and eigenvectors to plot them with a second code.
For Hydrogen is important to set lam_type="SA" because the interesting eigvalues are negative
"""
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import kron, diags, eye


def U(x, y, z, pot):
    '''
    Potetntial

    Parameters
    ----------
    x, y : ndarray like
        grid from np.meshgrid
    pot : string
        'hydrogen' or 'harmonic'
    '''
    if pot == "harmonic":
        return 0.5*( x**2 + y**2 + z**2)
    if pot == "hydrogen":
        return -1/np.sqrt(x**2 + y**2 + z**2)


def diag_H(N, k, dx, lam_type, pot):
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
    lam_type : string
        type of eigenvalues to find, smaller algebraic or in magnitude
    pot : string
        type of poptential, se U funnction
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
    V   = diags(U(X, Y, Z, pot).reshape(N**3), 0)
    H   = -1/(2*dx**2)*P + V

    eigval, eigvec = eigsh(H, k=k, which=lam_type)

    return eigval, eigvec

#=========================================================
# Computational parameters
#=========================================================

N        = 150                       # number of points
pot      = 'hydrogen'                # type of potential
lam_type = 'SA'                      # which eigenvalues find
x        = np.linspace(-20, 20, N)   # grid
X, Y, Z  = np.meshgrid(x, x, x)      # 3D grid
dx       = np.diff(x)[0]             # size step
k        = 14                        # number of eigvalues

start = time.time()
eigval, eigvec = diag_H(N, k, dx, lam_type, pot)
print(f"{time.time()-start} s")

# Save data
np.savez(f"data_{pot}_{N}", eigval, eigvec)

#=========================================================
# Quickly comparison to se if all is ok
#=========================================================

deg = lambda n: int(0.5*(n + 1)*(n + 2)) # degeneracy of oscillator

# True energetic level
if pot == "harmonic":
    E_n = [(n + 3/2) for n in range(4) for g in range(deg(n))]
if pot == "hydrogen":
    E_n = [-1/(2*n**2) for n in range(1, 4+1) for l in range(0, n) for m in np.arange(-l, l+1)]

for i in range(len(eigval)):
    print(f"{eigval[i]:.5f}, {E_n[i]}")
