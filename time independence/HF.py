"""
Code for the restricted hartree fock method.
The two examples to test this are helium and beryllium
"""
import time
import numpy as np
from scipy.linalg import eigh

HARTREE_TO_EV = 27.211

def P_matrix(coeff, N):
    """
    Compute density matrix P.

    Parameters
    ----------
    coeff : 2darray
        coefficents matrix
    N : int 
        number of electrons
    
    Returns
    -------
    P : 2darray
        density matrix
    """
    M = coeff.shape[0]
    P = np.zeros([M, M])
    
    for i in range(M):
        for j in range(M):
            for k in range(int(N/2)):
                P[i, j] += 2 * coeff[i, k] * coeff[j, k]

    return P


def G_matrix(P, R):
    """
    Compute G matrix.
    G = coulombic repulsion energy + exchange energy

    Parameters
    ----------
    P : 2darray
        density matrix
    R : 2darray
        electron repulsion matrix
    
    Returns
    -------
    G : 2darray
        repulsion matrix
    """
    num_bfs = P.shape[0]
    G = np.zeros((num_bfs, num_bfs))

    for i in range(num_bfs):
        for j in range(num_bfs):
            g = 0
            for k in range(num_bfs):
                for l in range(num_bfs):
                    int1 = R[i, j, k, l]
                    int2 = R[i, l, k, j]
                    g += P[k, l] * (int1 - 0.5 * int2)
            G[i, j] = g

    return G


def F_matrix(H, G):
    """
    Compute fock matrix F.
    F =  H_core + G

    Parameters
    ----------
    H : 2darray
        Hamiltonian core matrix
    G : 2darray
        repulsion matrix
    
    Returns
    -------
    F : 2darray
        fock matrix
    """
    return H + G



def diag_AB(A, B):
    """
    Function tha find all v and λ that satisfies:
    
    A @ v = λ * B @ v

    Parameters
    ----------
    A, B : 2darray

    Returns
    -------
    eigval : 1darray
        eigenvalues
    eigvec : 2darray
        eigenvectors
    """
    eigval, eigvec = eigh(A, B)
    return eigval, eigvec


def tot_ene(e, N, P, H, Vnn=0):
    """
    Compute the total energy.

    Parameters
    ----------
    e : 1darray
        energies
    N : int
        num of electrons
    P : 2darray
        density matrix
    H : 2darray
        H_core matrix
    Vnn : float
        nuclear nuclear repulsion energy, for atom is 0
    """
    e_tot = 0
    for i in range(int(N/2)):
        e_tot += e[i].real
    e_tot = e_tot + 0.5 * (P * H).sum() + Vnn
    return e_tot


def HF(path, Z, tol, verbose=False):
    """
    Run restricted hartree fock for a single atom.

    Parameters
    ----------
    path : string
        path of the file created by HF_integrals.py
    Z = int
        number of electrons
    tol : float
        required tollerance
    verobse : bool, optional default False
        if True print energy at each iteration
    
    Return
    ------
    hf_e : float
        energy of the groud state
    """

    N = Z  # num of electron = nuclear charege (since it's atom)

    MAT = np.load(path)
    H = MAT["H"]
    S = MAT["S"]
    R = MAT["R"]

    ene, coeff = diag_AB(H, S)
    
    P    = P_matrix(coeff, N)
    Vnn  = 0  # No nuclear repulsion for a single atom
    hf_e = tot_ene(ene, N, P, H, Vnn)

   
    delta_e  = 1
    count    = 0
    hf_e_old = hf_e

    # Iterations
    while(delta_e > tol):

        G = G_matrix(P, R)
        F = H + G

        ene, coeff = diag_AB(F, S)
        
        P    = P_matrix(coeff, N)
        hf_e = tot_ene(ene, N, P, H, Vnn)

        delta_e  = np.abs(hf_e - hf_e_old)
        hf_e_old = hf_e
        count   += 1

        if verbose:
            print(f"iter: {count}, \t energy: {hf_e}, \t dE : {delta_e}")
        
    return hf_e

def test():
  
    start = time.time()

    print("Test Helium:")
    Z = 2
    ene_he = HF("HF_helium_integrals.npz", Z, 1e-6, verbose=True)
    ref_he = -2.8616726 # tabulated value
    
    print(f"Computed = {ene_he}")
    print(f"Expected = {ref_he}")
    print()

    print("Test Lithium:")
    Z = 3
    ene_li = HF("HF_lithium_integrals.npz", Z, 1e-10, verbose=True)
    ref_li = -7.43124 # tabulated value

    print(f"Computed = {ene_li}")
    print(f"Expected = {ref_li}")
    print()

    print("Test Beryllium:")
    Z = 4
    ene_Be = HF("HF_beryllium_integrals.npz", Z, 1e-10, verbose=True)
    ref_Be = -14.572369 # tabulated value

    
    print(f"Computed = {ene_Be}")
    print(f"Expected = {ref_Be}")
    print()

    end = time.time() - start
    print(f"Elapsed time = {end:.3f} s")

if __name__ == "__main__":
    test()
