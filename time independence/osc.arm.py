"""
Code to solve the 1D Schrödinger equation by discretizing the Laplacian for the harmonic potential
"""
import time
import numpy as np
import scipy.special as ssp
from scipy.sparse import diags
import  matplotlib.pyplot  as  plt

start = time.time()

n  = 1000                    # number of points
xr = 10                      # right boundary
xl = -10                     # left boundary
L  = xr - xl                 # size of box
h  = (xr - xl)/n             # step size
tt = np.linspace(0, n, n)    # auxiliar array
xp = xl + h*tt               # grid on x

def U(x):
    """ harmonic potential
    """
    return 1/2 * x**2

def G(x, m):
    """ Analytical solution
    """
    return (1/(np.pi)**(1/4))*(1/np.sqrt((2**m)*ssp.gamma(m+1)))*ssp.eval_hermite(m, x)*np.exp(-(x**2)/2)
    
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
m = 10 # index for plot

plt.figure(1)
plt.title(f"$\psi(x)$ Harmonic oscillator n={m}", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)$', fontsize=15)
plt.grid()
plt.ylim(m-0.2,m+1.3)

plt.errorbar(xp, psi[:,m]+eigvals[m], fmt='.', label='$\psi(x)$ computed')
plt.plot(xp, G(xp, m)+eigvals[m], color='red', label='$\psi(x)$ exact')
plt.plot(xp, U(xp), color='black', label='V(x)= $ \dfrac{1}{2} x^2 $')
plt.plot(xp, np.ones(len(xp))*eigvals[m], color='blue', linestyle='--', label='$E_{%d}=%f$' %(m, eigvals[m]))
plt.legend(loc='best')

plt.figure(3)
plt.title("Error", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)_{num}-\psi(x)_{es}$', fontsize=15)
plt.grid()
plt.errorbar(xp, psi[:,m]-G(xp, m), fmt='.')

print(f"--- {time.time() - start} seconds ---")

plt.show()

print(eigvals[0:m]-(1/2 + np.linspace(0, m-1, m)))
