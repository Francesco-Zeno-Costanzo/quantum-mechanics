"""
Code to Plot the result of Schrodinger3D_solve.py
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

N       = 150                       # number of points
x       = np.linspace(-7, 7, N)     # grid
X, Y, Z = np.meshgrid(x, x, x)      # 3D grid
dx      = np.diff(x)[0]             # size step
k       = 10                        # number of eigvalues

data = np.load(f"data_{N}.npz", allow_pickle='TRUE')
eigval, eigvec = data["arr_0"], data["arr_1"]

deg = lambda n: int(0.5*(n + 1)*(n + 2))
psi = lambda n: eigvec.T[n].reshape((N, N, N))#/np.sqrt(dx**3)

E_n = [(n + 3/2) for n in range(4) for g in range(deg(n))]

for i in range(len(eigval)):
    print(f"{eigval[i]:.5f}, {E_n[i]}")

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

    # create all poits via cartesian product
    R3 = [str(xyz) for xyz in itertools.product(x, x, x)]
    # we use strings to avoid problem for np.random.choice

    # Sample coordinates according abs(psi)^2
    coord = np.random.choice(R3, size=M, replace=True, p=abs(psi)**2)
    # Delete parentesis
    coord = [c[1:-1].split(',') for c in coord]
    # Unpack corrdinates and convert from string to float
    x, y, z = np.array(coord).T
    x = np.array([float(i) for i in x])
    y = np.array([float(i) for i in y])
    z = np.array([float(i) for i in z])

    return x, y, z

n = 5
k = deg(n)
x_p, y_p, z_p = sampling(eigvec.T[n], x, 10000)

fig = plt.figure(2, figsize=(9,9))

ax  = fig.add_subplot(111, projection='3d')
ax.set_title("3D wave function", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)
#p = ax.scatter(x, y, z, c=P, cmap='plasma', s=P);fig.colorbar(p)
ax.scatter(x_p, y_p, z_p, s=2, alpha=0.5, c='blue')
#plt.tight_layout()

#=========================================================
# Plot Energies
#=========================================================

E = eigval

plt.figure(3)
plt.title("Energies", fontsize=15)
plt.ylabel(r"E/$\hbar \omega$", fontsize=15)
plt.plot(E, marker='.', linestyle='', color='black')
plt.grid()
plt.ylim(np.min(E)-1, np.max(E)+1)

[plt.axhline(nx + ny + nz + 3/2, color='b', linestyle='--')
for nx in range(0,2) for ny in range(0,2) for nz in range(0,2)]

plt.show()
