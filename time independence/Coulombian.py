"""
Code to solve the 1D Schrödinger equation by discretizing the Laplacian for the coulomb potential.
The code works poorly if l > 0
"""
import time
import math
import numpy as np
import scipy.special as ssp
from scipy.sparse import diags
import  matplotlib.pyplot  as  plt

start = time.time()

n  = 1000                    # number of points
xr = 25                      # right boundary
xl = 1e-10                   # left boundary
L  = xr - xl                 # size of box
h  = (xr - xl)/n             # step size
tt = np.linspace(0, n, n)    # auxiliar array
xp = xl + h*tt               # grid on x
l  = 0                       # potentential's parameter

def U(x):
    """ Potential
    """
    return -1/x - l*(l+1)/x**2

def G(x, n, l):
    """ Analytical solution
    """
    C = (2/(n**2))*np.sqrt((math.factorial(n-l-1))/math.factorial(n+l))
    return C*((2*x/n)**l)*np.exp(-x/n)*ssp.eval_genlaguerre(n-l-1, 2*l+1, 2*x/n)
    
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
m = 2 # index for plot

plt.figure(1)
plt.title("Radial $\psi(x)$ of hydrogen", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)$', fontsize=15)
plt.errorbar(xp, (psi[:,m])**2+eigvals[m], fmt='.',  label=f'$\psi(x)$ numerica {m}')
plt.plot(xp, np.ones(len(xp))*eigvals[m], color='blue', linestyle='--', label='$E_{%d}=%f$' %(m, eigvals[m]))

plt.plot(xp, U(xp), color='black', label='V(x)=-$\dfrac{1}{x}$')
plt.ylim(-1,0.75)

plt.plot(xp, (xp*G(xp, m, l))**2+eigvals[m], color='red', label='$\psi(x)$ analitica %ds' %m)
plt.grid()
plt.legend(loc='best')

plt.figure(2)
plt.title("Error", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)_{num}-\psi(x)_{es}$', fontsize=15)
plt.errorbar(xp, psi[:,m]**2-(xp*G(xp, m, l))**2, fmt='.')
plt.grid()

print(f"--- {time.time() - start} seconds ---")
plt.show()
