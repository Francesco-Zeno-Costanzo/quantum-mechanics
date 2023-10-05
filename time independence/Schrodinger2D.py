"""
Code to solve the Schrodinger equation in two spatial dimensions via diagonalization
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import kronsum, diags

N = 150                    # number of points
x = np.linspace(0, 1, N)   # grid 
X, Y = np.meshgrid(x, x)   # 2D grid
dx = np.diff(x)[0]         # size step

def U(x, y):
    """ Potenteial
    """
    return 0*X*Y

#=========================================================
# Build hamiltonian of system and diagonalizzation
# By discretizing the two spatial derivatives, since they
# are different coordinates, the tensor product between the
# matrices and the identity must be made:
# d^2/dx^2 x I_y + I_x x d^2/dy
# This is all implemented in the kronsum function
#=========================================================

P = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
P = -1/(2*dx**2) * kronsum(P, P)
V = diags(U(X, Y).reshape(N**2), 0)
H = P + V

eigval, eigvec = eigsh(H, k=10, which='SM')

psi = lambda n: eigvec.T[n].reshape((N,N))

#=========================================================
# Plot psi
#=========================================================

n = 4

fig = plt.figure(1)
ax  = fig.add_subplot(111)
ax.set_title("2D wave function", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_aspect('equal', 'box')
ax.contourf(X, Y, abs(psi(n))**2, 20, cmap='plasma')

fig = plt.figure(2)
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("2D wave function", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.plot_surface(X, Y, abs(psi(n))**2, cmap='plasma')

#=========================================================
# Plot E
# E_{n_x n_y} = \alpha (n_x^2 + n_y^2) so if n_x = n_y = 1
# we obtain \alpha = E_{11} / 2; we will plot E/\alpha to
# see the degeneracy trend  
#=========================================================

a = eigval[0]/2
E = eigval / a

plt.figure(3)
plt.title("Energies", fontsize=15)
plt.ylabel("E [a.u.]", fontsize=15)
plt.plot(E, marker='.', linestyle='', color='black')
plt.grid()
plt.ylim(np.min(E)-1, np.max(E)+1)
[plt.axhline(nx**2 + ny**2, color='b', linestyle='--') for nx in range(1,5) for ny in range(1,5)]

plt.show()
