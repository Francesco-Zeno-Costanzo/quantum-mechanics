"""

"""
import numpy as np
from scipy import signal
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def pot_mat_k(N_x, N_G, G_values, V_x, x): 
    '''
    Function to compute the potential matrix in reciprocal space.
    
    Parameters
    ----------
    N_x : int
        Number of points in the real space grid.
    N_G : int
        Number of reciprocal lattice vectors.
    G_values : 1darray_like
        List of reciprocal lattice vectors.
    V_x : 1darray_like
        Potential energy in real space.
    x : 1darray_like
        Real space grid.

    Returns
    -------
    V_mat : 2darray
        Potential matrix in reciprocal
    '''
   
    # Fourier transform of the potential
    V_G = np.array([np.sum(V_x * np.exp(-1j * G * x)) / N_x for G in G_values])
    
    # Initialize the potential matrix to ensure complex dtype
    V_mat = np.zeros((N_G, N_G), dtype=complex)
    
    for i, G_i in enumerate(G_values):
        for j, G_j in enumerate(G_values):
            G_diff = G_i - G_j
            idx = np.where(G_values == G_diff)[0]  
            if len(idx) > 0:
                V_mat[i, j] = V_G[idx[0]]
    """
    # Construct potential matrix using broadcasting
    G_diff = G_values[:, None] - G_values[None, :]
    idx_map = {G: i for i, G in enumerate(G_values)}
    
    # Vectorized lookup for elements
    idx = np.vectorize(idx_map.get)(G_diff, -1)
    mask = idx != -1
    V_mat = np.zeros((N_G, N_G), dtype=complex)
    V_mat[mask] = V_G[idx[mask]]
    """

    return V_mat


def hamiltonian(N_G, k, G_values, V):
    '''
    Function to compute the Hamiltonian matrix for a given k-point.

    Parameters
    ----------
    N_G : int
        Number of reciprocal lattice vectors.
    k : float
        k-point, value of crystall momentum.
    G_values : 1darray_like
        List of reciprocal lattice vectors.
    V : 2darray_like
        Potential matrix in reciprocal space.
    
    Returns
    -------
    H : 2darray
        Hamiltonian matrix for the given k.
    '''
    
    # Kinetic energy + potential energy
    H = np.diag((k + G_values)**2 / 2 ) + V
    
    return H


def compute_bands(N_x, N_G, k_points, V_x, x, a):
    '''
    Function to compute the energy bands of a 1D periodic potential.
    Returns the energy bands for each k-point in a matrix of shape (len(k_points), N_G).
    So the i-th column of the matrix contains the i-th energy band for all k-points.
    
    Parameters
    ----------
    N_x : int
        Number of points in the real space grid.
    N_G : int
        Number of reciprocal lattice vectors.
    k_points : 1darray_like
        List of k-points where to compute the bands.
    V_x : 1darray_like
        Potential energy in real space.
    x : 1darray_like
        Real space grid.
    a : float
        Lattice spacing.
    
    Returns
    -------
    ene : 2darray
        Energy bands for each k-point.
    '''
    
    # Reciprocal lattice vectors
    G_val = np.array([(2 * np.pi / a) * n for n in range(-N_G//2, N_G//2)])
    # Compute the potential matrix in reciprocal space
    V_k = pot_mat_k(N_x, N_G, G_val, V_x, x)
    
    # Compute the energy bands for each k-point
    ene = []
    for k in k_points:
        H_k     = hamiltonian(N_G, k, G_val, V_k)
        eigvals = eigh(H_k, eigvals_only=True)
        ene.append(eigvals)
    
    return np.array(ene)

def compute_gap(ene):
    gap = np.min(ene[:,1] - ene[:,0])
    return gap


L   = 100                      # Length of the system
N_x = 10000                    # Number of points in the real space grid
x   = np.linspace(0, L, N_x)   # Real space grid
V0  = 10                       # Potential amplitude
a   = 1                        # Lattice spacing
N_G = 100                      # Number of reciprocal lattice vectors
k   = np.linspace(-np.pi/a, np.pi/a, 1000) # crystal momentum

#==================== Define the potential ====================
epsilon = 0.01
V_x = 0
for i in np.arange(0, L, a):
    V_x +=  -1/np.sqrt((x - i)**2 + epsilon**2)

V_x = 0.5 * V_x

#V_x = V0 * (signal.square(2 * np.pi * x / a, duty=0.9)-1)
#plt.plot(x, V_x);plt.show();exit()

#==================== Compute the energy bands ====================

ene = compute_bands(N_x, N_G, k, V_x, x, a)
gap = compute_gap(ene)
print(f"Gap: {gap:.4f}")

#==================== Plot the energy bands ====================

plt.figure(figsize=(7, 5))
for i in range(min(4, ene.shape[1])):
    plt.plot(k, ene[:, i])

plt.xlabel(r"$k$", fontsize=14)
plt.ylabel(r"$E(k)$", fontsize=14)
plt.title("Struttura a bande con potenziale periodico")
plt.axvline(k[0], color='gray', linestyle='--')
plt.axvline(k[-1], color='gray', linestyle='--')
plt.grid()

plt.show()
"""
plt.figure(figsize=(7, 5))
NG = np.linspace(2, 100, 99, dtype=int)
all_gap = []
for N_G_test in NG:
    ene_test = compute_bands(N_x, N_G_test, np.linspace(-np.pi/a, np.pi/a, 100), V_x, x, a)
    gap = compute_gap(ene_test)
    all_gap.append(gap)

all_gap = np.array(all_gap)
plt.plot(NG, abs(all_gap-all_gap[-1]))

plt.xlabel(r"$N_G$")
plt.ylabel(r"first gap")
plt.title("Convergenza della prima banda al variare di $N_G$")
plt.grid()
plt.show()
"""