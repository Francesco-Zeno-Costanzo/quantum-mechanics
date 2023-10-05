import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def psi_inc(x):
    ''' Initial wave function
    '''

    x0 = -0.1      # center of gaussian 
    a  = 0.01      # variance
    k  = 200000    # wave number

    A = 1. #/ np.sqrt( 2 * np.pi * a**2 ) # normalizzation
    K1 = np.exp( - ( x - x0 )**2 / ( 2. * a**2 ) )
    K2 = np.exp( 1j * k * x )

    return A * K1 * K2

def U(x):
    ''' potential
    '''
    A = 1e6
    s = 0.001
    return A*np.exp(-(x/s)**2)


n  = 1001                  # number of point
a  = -0.5                  # left boundary
b  = -a                    # right boundary
x  = np.linspace(a, b, n)  # grid on x axis
dx = np.diff(x)[0]         # step size
T  = 0.0035                # Total time of evolution
dt = 1e-5                  # step size for time evolution
ts = int(T/dt)             # number of iteration

#=========================================================
# Build hamiltonian of system
#=========================================================

P = -1/(2*dx**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
V = sp.diags(U(x), 0, shape=(n, n))
H = P + V
#identity matrix
I =  sp.diags([1], 0, shape=(n, n))

#=========================================================
# Start of simulation
#=========================================================

psi = psi_inc(x)

PSI_T = np.zeros((ts, len(psi))) # abs(psi)^2
PSI_I = np.zeros((ts, len(psi))) # Im(psi)
PSI_R = np.zeros((ts, len(psi))) # Re(psi)

PSI_T[0, :] = abs(psi)**2
PSI_R[0, :] = np.real(psi)
PSI_I[0, :] = np.imag(psi)


fig = plt.figure()
plt.title("Gaussian packet propagation")
plt.plot(x, U(x), label='$V(x)$', color='black')
plt.grid()

plt.ylim(-0.1, np.max(PSI_T[0,:]))

# Time evolution
for i in range(ts):

    A = (I - 1j * dt/2. * H)
    b = (I + 1j * dt/2. * H) * psi

    psi = sp.linalg.spsolve(A,b)

    PSI_T[i, :] = abs(psi)**2
    PSI_R[i, :] = np.real(psi)
    PSI_I[i, :] = np.imag(psi)

#=========================================================
# Animation
#=========================================================

line0, = plt.plot([], [], 'b', label=r"$|\psi(x, t)|^2$")
#line1, = plt.plot([], [], 'r', label=r"$Im(\psi(x, t))$")
#line2, = plt.plot([], [], 'y', label=r"$Re(\psi(x, t))$")

def animate(i):
    line0.set_data(x, PSI_T[i, :])
    #line1.set_data(x, PSI_I[i, :])
    #line2.set_data(x, PSI_R[i, :])
    return line0, #line1, line2

plt.legend(loc='best')

anim = animation.FuncAnimation(fig, animate, frames=ts, interval=10, blit=True, repeat=True)

#anim.save('tunnel barriera.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
