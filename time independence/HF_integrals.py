"""
Code for calculating the integrals needed to use the Hartree Fock method.
Each integral is calculated in spherical coordinates,
so these results will only be valid for atoms.
Furthermore, since the angular part is not included (in the STO functions)
only atoms that have only one s orbital could be represented.
To conclude, the method used in the HF.py code is the restricted Hartree-Fock
so is only good in the case of closed shells.
"""
import time
import sympy as sp
import numpy as np


r, r1, r2, zeta = sp.symbols("r, r1, r2, zeta")
n = sp.Symbol('n', integer=True)


def STO(zeta, n, r=r):
    """
    Define a Slater Type Orbital function using sympy.

    Parameters
    ----------
    zeta : float
        zeta for the STO. Zeta is a constant related to
        the effective charge of the nucleus, i.e. the
        nuclear charge being partly shielded by electrons. 
    n : int
        principle quantum number for the STO.
    
    Retunr
    ------
    f : simpy function
        Normalize orbital functions
    """
    f = r ** (n - 1) * sp.exp(-zeta * r)
    # normalization
    N = sp.sqrt(sp.integrate(4 * sp.pi * f**2 * r**2,  (r, 0, +sp.oo)))
    return f / N

def H_core(f1, f2, Z):
    """
    Compute H_core integral between two STO functions.
    H_core = electron kinetics energy + electron nuclear potential energy

    Parameters
    ----------
    f1, f2 : sympy functions
        component of function basis
    Z : int
        Nuclear charge
    
    Returns
    -------
    H_core : float
        expetation value of H_core between f1 and f2 functions
    """
    T = - (1 / 2) * (1 / r) * sp.diff(r * f2, r, 2)
    V = - (Z / r) * f2
    H_core = sp.integrate(f1 * (T + V) * 4 * sp.pi * r**2 , (r, 0, +sp.oo))
    
    return H_core

def H_matrix(bfs, Z):
    """
    Compute the core hamiltonian matrix H.

    Parameters
    ----------
    bfs : list
        basis functions
    Z : int
        nuclear charge
    
    Returns
    -------
    H : 2darray
        core hamiltonian matrix
    """
    num_bfs = len(bfs)
    H       = np.zeros((num_bfs, num_bfs))

    for i in range(num_bfs):
        for j in range(num_bfs):
            H[i, j] = H_core(bfs[i], bfs[j], Z)

    return H

def overlap(f1, f2):
    """
    Compute overlap integral between two STO functions.

    Parameters
    ----------
    f1, f2 : sympy functions
        component of function basis
    
    Returns
    -------
    float
        Overlsp integral between f1 and f2
    """
    return sp.integrate(f1 * f2 * 4 * sp.pi * r**2, (r, 0, +sp.oo))

def S_matrix(bfs):
    """
    Compute overlap matrix S.

    Parameters
    ----------
    bfs : list
        basis functions
    
    Returns
    -------
    S : 2darray
        overlap matrix
    """
    num_bfs = len(bfs)
    S       = np.zeros((num_bfs, num_bfs))

    for i in range(num_bfs):
        for j in range(num_bfs):
            S[i, j] = overlap(bfs[i], bfs[j])

    return S

def two_el(four_bfs):
    """
    Compute electron-electron repulsion integral using the 1/r> approximation.
    The repulsion integral between two electrons is typically expressed as:

    ∫∫ (phi_1(r1) * phi_2(r1) * phi_3(r2) * phi_4(r2)) / |r1 - r2| dr1 dr2

    However, handling the term 1/|r1 - r2| directly is complicated due to
    the radial dependence. To simplify, we use the fact that:

    1 / |r1 - r2| = 1 / r>, where r> is the greater of r1 and r2.

    This allows us to split the integral into two regions:
    1) From 0 to r1 (where r1 > r2, hence 1/r1 is used),
    2) From r1 to infinity (where r2 > r1, hence 1/r2 is used).

    The final result is then integrated over r1 from 0 to infinity,
    giving the full electron-electron repulsion integral.

    Parameters
    ----------
    four_bfs : list
        list containing 4 basis functions

    Returns
    -------
    The value of the electron-electron repulsion integral.    
    """
    f1, f2, f3, f4 = four_bfs
    
    # Variables raname for integral
    f1 = f1.subs(r, r1)
    f2 = f2.subs(r, r1)
    f3 = f3.subs(r, r2)
    f4 = f4.subs(r, r2)

    I  = sp.integrate((1 / r1) * f3 * f4 * 4 * sp.pi * r2**2, (r2, 0,  r1))
    I += sp.integrate((1 / r2) * f3 * f4 * 4 * sp.pi * r2**2, (r2, r1, +sp.oo))
    return sp.integrate(f1 * f2 * 4 * sp.pi * r1**2 * I, (r1, 0, +sp.oo))

def R_matrix(bfs):
    """
    Compute the electron repulsion integral matrix R.
    The calculation of some elements could be avoided
    thanks to symmetries. We go by brute force and ignorance

    Parameters
    ----------
    bfs : list
        basis functions
    
    Returns
    -------
    R : 2darray
        Repulsion matrix
    """

    num_bfs = len(bfs)
    R = np.zeros((num_bfs, num_bfs, num_bfs, num_bfs))

    for i in range(num_bfs):
        for j in range(num_bfs):
            for k in range(num_bfs):
                for l in range(num_bfs):
                    R[i, j, k, l] = two_el([bfs[i], bfs[j], bfs[k], bfs[l]])


    return R

def helium():
    
    start = time.time()

    # Use 2 Slator Type ourbital to represent Helium 1s orbital.
    # The final Helium 1s orbital is a linear combination of these two STO.    
    f1s_1 = STO(1.45363, n=1)
    f1s_2 = STO(2.91093, n=1)

    phi_i = [f1s_1, f1s_2]

    Z = 2
    print("Computing H ...")
    H = H_matrix(phi_i, Z)
    print("Computing S ...")
    S = S_matrix(phi_i)
    print("Computing R ...") # takes time
    R = R_matrix(phi_i)

    np.savez('HF_helium_integrals', H=H, S=S, R=R)

    end = time.time() - start
    print(f"Elapsed time = {end:.3f} s") # 10.537 s

def beryllium():
    
    start = time.time()

    # Use 2 STO to represent Be 1s orbital and another 2 STO for 2s orbital
    # The final 1s orbital is a linear combination of these 4 STO.
    # Same for 2s orbital.
    f1s_1 = STO(5.59108, n=1)
    f1s_2 = STO(3.35538, n=1)
    f2s_1 = STO(1.01122, n=2)
    f2s_2 = STO(0.61000, n=2)

    phi_i = [f1s_1, f1s_2, f2s_1, f2s_2]

    Z = 4
    print("Computing H ...")
    H = H_matrix(phi_i, Z)
    print("Computing S ...")
    S = S_matrix(phi_i)
    print("Computing R ...") # takes time
    R = R_matrix(phi_i)

    np.savez('HF_beryllium_integrals', H=H, S=S, R=R)

    end = time.time() - start
    print(f"Elapsed time = {end:.3f} s") # 183.967 s



if __name__ == "__main__":
    helium()
    beryllium()