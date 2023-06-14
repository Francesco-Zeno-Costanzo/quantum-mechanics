"""
Code to solve schrodinger equation for finite well via im shooting method.
"""
import time
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


def V(x, x0, V0, l):
    '''
    potential
    x : position
    '''
    V = 0
    if x > x0 or x == 0:
        V = 0
    elif x < x0:
        V = V0
  
    return V - l*(l+1)/x**2


def F(x, E, x0, V0, l) :
    
    dydt = 2*(V(x, x0, V0, l) - E)
    
    return dydt

def Numerov(num_steps, xi, xf, init_f, init_b, f, args=()):
    """
    Integrator with Adams-Bashforth-Moulton
    predictor and corretor of order 4

    Parameters
    ----------
    num_steps : int
        number of point of solution
    tf : float
        upper bound of integration
    f : callable
        function to integrate, must accept vectorial input
    init : 1darray
        array of initial condition
    args : tuple, optional
        extra arguments to pass to f

    Return
    ------
    X : array, shape (num_steps + 1, len(init))
        solution of equation
    t : 1darray
        time
    """
    #time steps
    dx = (xf - xi)/num_steps

    y = np.zeros(num_steps + 1)    #matrice delle soluzioni

    y[0],  y[1]  = init_f              #condizioni iniziali
    y[-1], y[-2] = init_b              #condizioni iniziali
    
    x     = np.array([xi + i*dx for i in range(num_steps + 1)])
    fact  = np.array([dx**2 / 12 * f(z, *args) for z in x])
    index = 0
    for i in range(num_steps):
        if fact[i] != np.copysign(fact[i], fact[i+1]):
            index = i
    
    fact = 1 - fact
    
    for i in range(1, index):
        num = (12 - 10*fact[i])*y[i] - fact[i - 1]*y[i - 1]
        den = fact[i + 1]
        y[i + 1] = num/den
    
    y_m = y[index]
    
    for i in range(num_steps - 1, index, -1):
        num = (12 - 10*fact[i])*y[i] - fact[i + 1] * y[i + 1]
        den = fact[i - 1]
        y[i - 1] = num/den
    
    y_m /= y[index]
    
    for i in range(index, num_steps + 1):
        y[i] *= y_m
    
    y = y/np.sqrt(si.simps(np.square(y), x, even='first'))
    
    jump = (y[index + 1] + y[index - 1] - (14 - 12*fact[i]) * y[index]) / dx
    jump = jump * y[index]

    return x, y, jump


def shooting(num_steps, x_min, x_max, init_guess, step, tol, args=()):
    
    dx = (x_max - x_min)/num_steps
    E1 = init_guess
    
    init_f = np.array([0, dx])
    init_b = np.array([np.exp(E1*x_max), np.exp(E1*(num_steps-1)*dx)])
    
    x1, y1, jump1 = Numerov(num_steps, x_min, x_max, init_f, init_b, F, args=(E1, *args))
    print(f'bound_1 : jump = {jump1:.5f}, ene = {E1:.5f}')
    
    while True:
        E2 = E1 + step
        x2, y2, jump2 = Numerov(num_steps, x_min, x_max, init_f, init_b, F, args=(E2, *args))

        if (jump1 * jump2) < 0.0:
            break
        jump1 = jump2
        E1 = E2
    print(f'bound_2 : jump = {jump2:.5f}, ene = {E2:.5f}')
    
    while True:
        
        Em = (E1 + E2)/2.0     
        xm, ym, jumpm = Numerov(num_steps, x_min, x_max, init_f, init_b, F, args=(Em, *args))
        
        if (jumpm * jump1) > 0 :
            jump1 = jumpm
            E1 = Em
        else :
            jump2 = jumpm
            E2 = Em
        
        R1 = abs(E1 - E2)
        R2 = abs(jumpm)
        print(f'jump = {R2:.3e}, dE = {R1:.3e}')
            
        if R2 < tol or R1 < tol :
            break   
    
    return xm, ym, Em, jumpm    

#================================================================================
# Computational parameter
#================================================================================


x_max = 30       # max distance
x_min = 1e-10    # min distance to avoid divergence in l*(l+1)/x**2
steps = 3000     # Number of point
step  = 0.001    # step for searching of second bound
tau   = 1.0e-12  # tollerance for bisection
x_b   = 10       # size of box
V0    = -1       # potential
E1    = -0.999   # initial guess for eigenvalue
l     = 0        # angular momentum


#================================================================================
# Computation
#================================================================================

start = time.time()

x, psi, eigv, jump = shooting(steps, x_min, x_max, E1, step, tau, args=(x_b, V0, l))

end = time.time() - start

print(f"E = {eigv:.10f}")
print(f"Elapsed time: {end:.3f}")


#================================================================================
# Plot
#================================================================================

plt.figure(1)
plt.title(f"$\psi(x)$ finite well", fontsize=20)
plt.plot(x, abs(psi)**2 + eigv, 'b', label='$\psi(x)$ numerical' )
plt.plot(x, np.ones(len(x))*eigv, color='black', linestyle='--', label='$E=%f$' %eigv)
plt.plot(x, [V(z, x_b, V0, l) for z in x], color='red', label='V(x)')
plt.ylim(V0-0.1, 0.2)
plt.grid()
plt.legend(loc='best')
plt.ylabel('$\psi(x)$', fontsize=15)
plt.xlabel('x', fontsize=15)

plt.show()













