"""
Code to Plot the result of Schrodinger3D_solve.py
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

N       = 150                       # number of points
pot     = 'harmonic'                # type of potential
x       = np.linspace(-7, 7, N)   # grid

data = np.load(f"data_{pot}_{N}.npz", allow_pickle='TRUE')
eigval, eigvec = data["arr_0"], data["arr_1"]

#=========================================================
# Plot psi
#=========================================================

def sampling(psi, x, M):
    '''
    Function to crate the plot of psi. Beeing impossible to
    plot a function from R^3 to R here we use that abs(psi)^2
    is a probability density. So after create all possible points
    for our box we resample them according abs(psi)^2.
    These points are the ones that will be plotted.

    Parameters
    ----------
    psi : 1darray
         wave function len(psi) = N**3
    x : 1darray
        grid of our box
    M : int
        number of poits that we want to plot

    Returns
    -------
    x, y, z : 1darray
        resampled coordinates
    '''
    N = x.size
    # create all points via cartesian product
    R3    = list(itertools.product(x, x, x))
    index = np.linspace(0, N**3, N**3, dtype=int)

    # Sample index coordinates according abs(psi)^2
    index_coord = np.random.choice(index, size=M, replace=True, p=abs(psi)**2)
    coord       = [R3[c] for c in index_coord]
    # Unpack corrdinates
    x, y, z = np.array(coord).T

    return x, y, z

n = 0
x_p, y_p, z_p = sampling(eigvec.T[n], x, 10000)

fig = plt.figure(2, figsize=(9,9))

ax  = fig.add_subplot(111, projection='3d')
ax.set_title("3D wave function", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)
ax.scatter(x_p, y_p, z_p, s=2, alpha=0.5, c='blue')
#plt.tight_layout()

#=========================================================
# Plot Energies
#=========================================================

E = eigval

plt.figure(3)
plt.title("Energies", fontsize=15)
plt.ylabel(r"E [n.u.]", fontsize=15)
plt.plot(E, marker='.', linestyle='', color='black')
plt.grid()


if pot == "harmonic":
    [plt.axhline(nx + ny + nz + 3/2, color='b', linestyle='--')
     for nx in range(0,2) for ny in range(0,2) for nz in range(0,2)]
    plt.ylim(np.min(E)-1, np.max(E)+1)

if pot == "hydrogen":
    [plt.axhline(-1/(2*n**2), color='b', linestyle='--') for n in range(1,4)]
    plt.ylim(-0.65, 0.1)

plt.show()
