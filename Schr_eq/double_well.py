"""
Code to solve the 1D Schrödinger equation by discretizing the Laplacian for the double well potential
"""
import time
import numpy as np
from scipy.sparse import diags
import  matplotlib.pyplot as plt

start = time.time()

n  = 1000                    # number of points
xr = 10                      # right boundary
xl = -10                     # left boundary
L  = xr - xl                 # size of box
h  = (xr - xl)/n             # step size
tt = np.linspace(0, n, n)    # auxiliar array
xp = xl + h*tt               # grid on x
g  = 0.05                    # parameter of potential

def U(x):
    """ Double well potential
    """
    return -1/2 * x**2 + (g/2)*x**4 + 1/(8*g)

#=========================================================
# Build hamiltonian of system and diagonalization
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
m = 1 # index for plot

plt.figure(1)
plt.title(f"$\psi(x)$ double well n={m}", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)$', fontsize=15)
plt.grid()
plt.ylim(np.min(psi[:,m]+eigvals[m])-0.2, np.max(psi[:,m]+eigvals[m])+0.2)

plt.errorbar(xp, psi[:, m] + eigvals[m], fmt='.', label='$\psi(x)$')
plt.plot(xp, U(xp), color='black', label='V(x)= $- \dfrac{1}{2} x^2 +  \dfrac{g}{2} x^4 + \dfrac{1}{8g}$')
plt.plot(xp, np.ones(len(xp))*eigvals[m], color='black', linestyle='--', label='$E_{%d}=%f$' %(m, eigvals[m]))
plt.legend(loc='best')


plt.show()
print(f"--- {time.time() - start} seconds ---")

