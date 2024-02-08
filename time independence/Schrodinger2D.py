"""
Code to solve the Schrodinger equation in two spatial dimensions via diagonalization
we have implemented harmonic ad infinite well potential
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse import kronsum, diags


def U(x, y, pot):
    '''
    Potential

    Parameters
    ----------
    x, y : ndarray like
        grid from np.meshgrid
    pot : string
        'well' or 'harmonic'
    '''
    if pot == "well":
        return 0*x
    if pot == "harmonic":
        return 0.5*( x**2 + y**2 )


def diag_H(N, k, dx, pot):
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
    pot : string
        type of poptential, se U funnction
    '''
    #=========================================================
    # Build hamiltonian of system and diagonalization
    # By discretizing the two spatial derivatives, since they
    # are different coordinates, the tensor product between the
    # matrices and the identity must be made:
    # d^2/dx^2 x I_y + I_x x d^2/dy
    # This is all implemented in the kronsum function
    #=========================================================

    P = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
    P = -1/(2*dx**2) * kronsum(P, P)
    V = diags(U(X, Y, pot).reshape(N**2), 0)
    H = P + V

    eigval, eigvec = eigsh(H, k=10, which='SM')

    return eigval, eigvec

#=========================================================
# Computational parameters
#=========================================================

N    = 150                    # number of points
pot  = 'harmonic'             # type of potential
#x    = np.linspace(0, 1, N)  # grid for infinite well
x    = np.linspace(-7, 7, N)  # grid for oscillator
X, Y = np.meshgrid(x, x)      # 2D grid
dx   = np.diff(x)[0]          # size step
k    = 10                     # number of eigvalues

eigval, eigvec = diag_H(N, k, dx, pot)
psi = lambda n: eigvec.T[n].reshape((N,N))/dx

#=========================================================
# Plot psi
#=========================================================

n = 8

fig = plt.figure(1, figsize=(8, 8))
plt.suptitle("2D wave function", fontsize=15)
for i in range(1, len(eigval)):
    ax = fig.add_subplot(3, 3, i)
    ax.set_aspect('equal', 'box')
    ax.contourf(X, Y, abs(psi(i))**2, 20, cmap='plasma')

fig = plt.figure(2)
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("2D wave function", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.plot_surface(X, Y, abs(psi(n))**2, cmap='plasma')

#=========================================================
# Plot E to see degeneracy
#=========================================================

# to see the dipendence from n
a = np.pi**2/2 if pot=='well' else 1
E = eigval / a

plt.figure(3)
plt.title("Energies", fontsize=15)
plt.ylabel(r"E/$\hbar \omega$", fontsize=15)
plt.plot(E, marker='.', linestyle='', color='black')
plt.grid()
plt.ylim(np.min(E)-1, np.max(E)+1)

if pot == "well":
    [plt.axhline(nx**2 + ny**2, color='b', linestyle='--') for nx in range(1,5) for ny in range(1,5)]
if pot == "harmonic":
    [plt.axhline(nx + ny, color='b', linestyle='--') for nx in range(0,5) for ny in range(0,5)]

plt.show()
