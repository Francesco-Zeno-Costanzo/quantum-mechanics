"""
Code to plot the results of matrix_free.f90
"""
import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.special import eval_hermite

eigval = np.loadtxt('mf_val.dat', unpack=True)
psi_x  = np.loadtxt('mf_vec.dat')
psi = psi_x[:-1, :]
xp  = psi_x[-1,  :]
psi = psi.T

def G(x, m):
    return (1/(np.pi)**(1/4))*(1/np.sqrt((2**m)*factorial(m)))*eval_hermite(m, x)*np.exp(-(x**2)/2)

def Potential(x):
    return 0.5 * x**2

plt.figure(1)
plt.title("Oscillatore armonico")#, fontsize=15)
plt.xlabel('x')#, fontsize=15)
plt.ylabel(r'$|\psi(x)|^2$')#, fontsize=15)
plt.grid()
plt.ylim(0, 3)
plt.xlim(-5, 5)

plt.plot(xp, Potential(xp) , color='black', label='V(x)= $ x^2/2 $')
c = ['b', 'r', 'g']

for L in range(3):
    
    plt.errorbar(xp, abs(psi[:,L])**2 + eigval[L], color=c[L], fmt='.')
    plt.plot(xp, np.ones(len(xp))*eigval[L], color=c[L], linestyle='--', label='$E_{%d}=%f$' %(L, eigval[L]))
    plt.plot(xp, abs(G(xp, L))**2 + eigval[L], color='k')
    
#plt.legend(loc='best')

plt.figure(2)
plt.title("$\psi(x)$", fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('$\psi(x)$', fontsize=15)
plt.grid()

c = ['b', 'r', 'g']

for L in range(3):
    plt.errorbar(xp, abs(abs(psi[:,L])**2 - abs(G(xp, L))**2), color=c[L], fmt='.')

plt.show()