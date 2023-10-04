"""
Code to solve the 1D Schrödinger equation by discretizing the Laplacian for the Morse potential
"""
import time
import numpy as np
import scipy.special as ssp
from scipy.sparse import diags
import  matplotlib.pyplot  as  plt

start = time.time()

n  = 2000                    # number of points
xr = 10                      # right boundary
xl = 0                       # left boundary
L  = xr - xl                 # size of box
h  = (xr - xl)/n             # step size
tt = np.linspace(0, n, n)    # auxiliar array
xp = xl + h*tt               # grid on x
x0 = 1                       # potential parameter
B  = 152                     # potential parameter

def U(x):
    """ Morse potential
    """
    return B*(1 - np.exp(-(x-x0)))**2
    

def G(x, n):
    """ Analytical solution
    """
    l  = np.sqrt(2*B)
    t1 = np.sqrt(((ssp.gamma(n+1)*(2*l-2*n-1))/ssp.gamma(2*l-n)))
    t2 = (2*l*np.exp(-(x-x0)))**(l-n-1/2)
    t3 = np.exp(-1/2 *2*l*np.exp(-(x-x0)))
    t4 = ssp.eval_genlaguerre(n, 2*l-2*n-1, 2*l*np.exp(-(x-x0)))
    return t1 * t2 * t3 * t4

#=========================================================
# Build hamiltonian of system and diagonalizzation
#=========================================================

P = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()
V = diags(U(xp), 0, shape=(n, n)).toarray()
H = -(1/(2*h**2))*P + V

eigval, eigvec = np.linalg.eig(H)

eigvals = np.sort(eigval)
eigvecs = eigvec[:,eigval.argsort()]
psi     = eigvecs/np.sqrt(h)

#=========================================================
# Plot
#=========================================================
m = 14 # index for plot


plt.figure(1)
plt.ylim(0, B)
#plt.ylim(np.min(psi[:,m]+eigvals[m]), np.max(psi[:,m]+eigvals[m]))
plt.title(f"$\psi(x)$ Morse's potential n={m}", fontsize=20)
plt.ylabel('$\psi(x)$', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.grid()

plt.errorbar(xp, psi[:,m] + eigvals[m], fmt='.', label='$\psi(x)$ computed')
plt.plot(xp, G(xp, m) + eigvals[m], color='red', label='$\psi(x)$ analytical')
plt.plot(xp, U(xp), color='black', label='V(x)= $B(1-e^{-(x-x_0)})^2$')
plt.plot(xp, np.ones(len(xp))*eigvals[m], color='black', linestyle='--', label='$E_{%d}=%f$' %(m, eigvals[m]))
plt.legend(loc='best')


plt.figure(3)
plt.title("Error", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)_{num}-\psi(x)_{es}$', fontsize=15)
plt.errorbar(xp, psi[:,m]-G(xp, m), fmt='.')
plt.grid()

print(f"--- {time.time() - start} seconds ---")
plt.show()
# True eigenvalues
E = np.array([])
N = int(np.sqrt(2*B)-0.5)
for i in range(N):
    o  = np.sqrt(2*B)
    En = o*(i+1/2)-((o**2)/(4*B))*(i+1/2)**2
    E  = np.insert(E, len(E), En)

print(eigvals[0:N]-E)

