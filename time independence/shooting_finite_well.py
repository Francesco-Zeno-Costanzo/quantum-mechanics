"""
Code to solve schrodinger equation for harmonic or anharmonic oscillator via im shooting method.
To solve the equation at each step we use Adams-Bashforth-Moulton predictor and corretor of order 4.
The resolution is done along a single direction and for x>0 then using the symmetry it extends to x<0.
"""
import time
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


def V(x, x0, V0):
    '''
    potential
    x : position
    '''
    V = 0
    if x > x0:
        V = 0
    elif x > -x0 and x < x0:
        V = V0
    elif x < x0:
        V = 0
    return V


def F(x, y, E, x0, V0) :
    
    psi, dpsi = y
    dydt = np.array([dpsi , 2*(V(x, x0, V0) - E)*psi ])
    
    return dydt


def AMB4(num_steps, tf, f, init, args=()):
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
    dt = tf/num_steps

    X = np.zeros((num_steps + 1, len(init))) #matrice delle soluzioni
    t = np.zeros(num_steps + 1)              #array dei tempi

    X[0, :] = init                           #condizioni iniziali

    #primi passi con runge kutta
    for i in range(3):
        xk1 = f(t[i], X[i, :], *args)
        xk2 = f(t[i] + dt/2, X[i, :] + xk1*dt/2, *args)
        xk3 = f(t[i] + dt/2, X[i, :] + xk2*dt/2, *args)
        xk4 = f(t[i] + dt, X[i, :] + xk3*dt, *args)
        X[i + 1, :] = X[i, :] + (dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        t[i + 1] = t[i] + dt

    # Adams-Bashforth-Moulton
    i = 3
    AB0 = f(t[i  ], X[i,   :], *args)
    AB1 = f(t[i-1], X[i-1, :], *args)
    AB2 = f(t[i-2], X[i-2, :], *args)
    AB3 = f(t[i-3], X[i-3, :], *args)

    for i in range(3,num_steps):
        #predico
        X[i + 1, :] = X[i, :] + dt/24*(55*AB0 - 59*AB1 + 37*AB2 - 9*AB3)
        t[i + 1] = t[i] + dt
        #correggo
        AB3 = AB2
        AB2 = AB1
        AB1 = AB0
        AB0 = f(t[i+1], X[i + 1, :], *args)

        X[i + 1, :] = X[i, :] + dt/24*(9*AB0 + 19*AB1 - 5*AB2 + AB3)

    return X, t
   

def shooting(x, n, init_eigv, step, tau, F, args=()):
    '''
    Solution found by shooting method, in only one direction.
    We use the simmetry for potential, so we have the same
    simmetry on psi and we solve the equation only for positive
    value for x. So for even energy levels psi(x) is even and
    the value on zero will be constant, so we set initial condition
    psi(0) = 1, d/dx psi(x) = 0.
    On the other hand if n is odd we set psi(0) = 0, d/dx psi(x) = 1
    
    Parameters
    ----------
    x : 1darray
        array of pososition
    n : int
        energy level
    init_eigv : float
        initial guess for eigenvalue
    step : float
        step for searching of zeros
    tau : float
        tollerance for searching
    F : callable
        equation to solve
    args : tuple, optional
        extra argument to pass at F
    
    Returns
    -------
    Em = float
        eigenvalue
    psi : 1darray
        half wave function
    '''
    
    # fisrt bound
    psi = []           # list for the final solution
    E1 = init_eigv     # initial guess
    if n%2 == 0:         
        y = np.array([1.0, 0.0]) # initial condition for even function
    else:
        y = np.array([0.0, 1.0]) # initial condition for odd  function
        
    #sol = si.odeint(F, y, x, args=(E1, *args), tfirst=True)
    sol, t = AMB4(N, x_max, F, y, args=(E1, *args))
    psi_tail_1 = sol[-1, 0] # keep the last vaule of psi
    # this value can be very wrong, beacuse we don't know the true
    # value of E, but the import thing is the sign.
    
    # second bound
    while True:
        E2 = E1 + step
        #sol = si.odeint(F, y, x, args=(E2, *args), tfirst=True)
        sol, t = AMB4(N, x_max, F, y, args=(E2, *args))
        psi_tail_2 = sol[-1, 0]

        if (psi_tail_1 * psi_tail_2) < 0.0:
            # if the two tail have differente signs we can 
            # start the research of zeros via bisection method
            break
        psi_tail_1 = psi_tail_2
        E1 = E2
     
    # Searching of the true eigevalue via bisection method
    while True:
        
        Em = (E1 + E2)/2.0
        if abs(E1 - E2) < tau :
            break
                
        #sol = si.odeint(F, y, x, args=(Em, *args), tfirst=True)
        sol, t = AMB4(N, x_max, F, y, args=(Em, *args))
        psi_tail_m = sol[-1, 0]
        
        if (psi_tail_m * psi_tail_1) > 0 :
            psi_tail_1 = psi_tail_m
            E1 = Em
        else :
            psi_tail_2 = psi_tail_m
            E2 = Em
    
    for i in range(len(x)) :
        psi.append(sol[i, 0]) # keep the solution
    
    return Em, psi


def true_psi(psi, x):
    '''
    the shooting function only returns half of the function
    we use the symmetry properties to complete it.
    
    Parameter
    ---------
    psi : list
        half psi not normalized
    x : 1darray
        total space
     
    Return
    ------
    psi : 1darray
        total and normalized wave function
    '''

    while True:
        if abs(psi[-2]) > abs( psi[-1]):
            break # the vaule of energy is precise but always with error
        psi.pop() # so it is possible that the tail of psi are wrong
          
    if len(psi) < (N+1):
        while len(psi) < (N+1):
            psi.append(0.0) # we set to zero to readjust the length
    
    # psi for negative x         
    psi_r = list(reversed(psi))
    if n%2 == 1: #  for n odd psi is odd
        for k in range(len(psi_r)):
            psi_r[k] = - psi_r[k]

    psi_r.pop()
    psi = psi_r + psi  # total function
    Norm = np.sqrt(si.simps(np.square(psi), x, even='first'))
    psi = np.array(psi)/Norm # nomalization
    
    return psi

#================================================================================
# Computational parameter
#================================================================================

N     = 1000                                   # Number of point
x_max = 30                                     # half size of box
steps = 0.005                                  # step for searching of second bound
tau   = 1.0e-12                                # tollerance for bisection
x     = np.linspace(0, x_max, N)               # array of half box, for solving eq in shooting
z     = np.linspace(-x_max, x_max, 2*N + 1)    # total array of distance in the box
x_b   = 10                                     # half width
V0    = -1                                     # potential
n     = 0                                      # "label of energy level" in reality it must be set either even or odd, the important thing is the energy guess
i_eig = -0.7                                   # initial guess for eigenvalue


#================================================================================
# Computation
#================================================================================

start = time.time()

eigv, psi = shooting(x, n, i_eig ,steps, tau, F, args=(x_b, V0))
psi = true_psi(psi, z)

end = time.time() - start

print(f"E_{n} = {eigv:.10f}")
print(f"Elapsed time: {end:.3f}")

#================================================================================
# Plot
#================================================================================

plt.figure(1)
plt.title(f"$\psi(x)$ finite well n={n}", fontsize=20)
plt.plot(z, psi + eigv, 'b', label='$\psi(x)$ numerical' )

plt.plot(z, np.ones(len(z))*eigv, color='black', linestyle='--', label='$E_{%d}=%f$' %(n, eigv))
plt.plot(z, [V(x, x_b, V0) for x in z], color='red', label='V(x)')
plt.ylim(V0-0.1, 0.2)
plt.grid()
plt.legend(loc='best')
plt.ylabel('$\psi(x)$', fontsize=15)
plt.xlabel('x', fontsize=15)

plt.show()

