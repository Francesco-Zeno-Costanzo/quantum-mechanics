"""
Code for the solution of Schrodinger's time dependent equation
via split operator method but unlike tunnel_barrier.py using a FFT.
Now the idea is tu use:
exp(1j (T+V) dt) = exp(1j V dt/2) exp(1j T dt) exp(1j V dt/2) + O(dt^3)
and to compute exp(1j T dt) we go in momentum space where T is diagonal
so it easy to compute, so we must use FFT to go from x space to p space
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#=========================================================
# Build the propagator in x and k space and time evolution
#=========================================================

def evolution(pot, init_psi, x, k, dt, time=1):
    """
    Compute evolution via split operator method.
    Is possible to evolve in real or immaginary time.
    
    Parameters
    ----------
    pot : callable
        potential in Schrodinger's equation
    init_psi : 1darray
        initial condition
    x : 1darray
        grid on space
    k : 1darray
        grid on momentum
    dt : float
        time step of evolution
    time : int or complex, optional, default 1
        time must be 1 or -1j, if time=1 we evolve in real time
        if istead time is -1j we obtein the ground state
    
    Return
    ------
    PSI : list
        list of all temporal evolution of psi
    """
    
    if time not in [1, -1j]:
        msg = """The variable time must be 1 for time evolution
            or -1j for evolution in immaginary time for ground state"""
        raise ValueError(msg)
    
    # grid spacing
    dx = np.diff(x)[0]
    
    # Propagator 
    U_x = np.exp(-1j * time * pot(x) * dt/2)  # Half step in space
    U_k = np.exp(-1j * time * k**2/2 * dt  )  # Full step in momentum
    
    PSI = []
    psi = init_psi # initial condition
    PSI.append(psi)
    
    # Time evolution
    for _ in range(ts):
        psi = U_x * psi
        
        psi_k = np.fft.fft(psi)
        psi_k = U_k * psi_k
        
        psi = np.fft.ifft(psi_k)
        psi = U_x * psi
        
        if time == -1j:
            norm = sum(abs(psi)**2) * dx
            psi  = psi/np.sqrt(norm)
        
        PSI.append(psi)
    
    return PSI

#=========================================================
# Compute the energy
#=========================================================

def energy(pot, psi, x, k):
    """
    Function to compute the energy
    
    Parameters
    ----------
    pot : callable
        potential in Schrodinger's equation
    init_psi : 1darray
        initial condition
    x : 1darray
        grid on space
    k : 1darray
        grid on momentum
    
    Return
    ------
    E : float
        total energy
    """
    
    dx = np.diff(x)[0]
    
    psi_k = np.fft.fft(psi)
    psi_c = psi.conj()
    
    T = 0.5 * psi_c * np.fft.ifft(k**2*psi_k)
    V = psi_c * pot(x) * psi
    
    E = sum(T.real + V.real) * dx
    
    return E

#=========================================================
# Initial wave function ad potential
#=========================================================

def U(x):
    ''' harmonic potential
    '''
    return 0.5*x**2

def psi_inc(x, x0, a, k):
    ''' Initial wave function
    '''
    A = 1. / np.sqrt( 2 * np.pi * a**2 ) # normalizzation
    K1 = np.exp( - ( x - x0 )**2 / ( 2. * a**2 ) )
    K2 = np.exp( 1j * k * x )
    # let's multiply by five so the animation is prettier
    return A * K1 * K2 * 5
        
#=========================================================
# Computational parameters
#=========================================================

n  = 1000                    # Number of points
xr = 10                      # Right boundary
xl = -xr                     # Left boundary
L  = xr - xl                 # Size of box
x  = np.linspace(xl, xr, n)  # Grid on x axis
dx = np.diff(x)[0]           # Space step
dt = 1e-3                    # Time step
T  = 10                      # Total time of simulation
ts = int(T/dt)               # Number of step in time
it = -1j                     # real or immaginary time, must be 1 or -1j
k  = np.fft.fftfreq(n, dx)   # Every possible value of momentum
k *= 2*np.pi 

#=========================================================
# Evolution and energy computation
#=========================================================

# Initializzation of gaussian wave packet
psi = psi_inc(x, -1.2, 0.5, 0.3)
PSI = evolution(U, psi, x, k, dt, time=it)

Ene = np.zeros(ts+1)

for i, psi in enumerate(PSI):
    Ene[i] = energy(U, psi, x, k)

#=========================================================
# Animation and plot
#=========================================================
plt.figure(0)
if it == -1j : 
    plt.plot(Ene-0.5, 'b')
    plt.ylabel('E(t) - E$_0$')
    plt.yscale('log')
if it == 1: 
    plt.plot(Ene-Ene[0], 'b')
    plt.ylabel('E(t) - E(0)')
plt.xlabel('time step')

plt.grid()

fig = plt.figure(1)
plt.title("Gaussian packet propagation")
plt.plot(x, U(x), label='$V(x)$', color='black')
plt.grid()

YLIM = np.max(abs(PSI[0])**2) if it == 1 else 1
plt.ylim(-0.0, YLIM)
plt.xlim(-5, 5)

line, = plt.plot([], [], 'b', label=r"$|\psi(x, t)|^2$")

def animate(i):
    line.set_data(x, abs(PSI[i])**2)
    return line,


anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, ts, 10), 
                               interval=1, blit=True, repeat=True)

#anim.save('ho.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

if it == -1j:
    print(f"Energy of ground state = {Ene[-1]}")
    
    def G(x):
        """ Analytical solution, ground state """
        return (1/(np.pi)**(1/4))*np.exp(-(x**2)/2)
    
    plt.plot(x, abs(G(x))**2, 'r', label='Analytical')

plt.legend(loc='best')
plt.show()

