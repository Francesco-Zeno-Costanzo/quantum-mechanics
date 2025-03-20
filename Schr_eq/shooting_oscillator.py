"""
Code to solve schrodinger equation for harmonic or anharmonic oscillator via im shooting method.
To solve the equation at each step we use odeint.
The resolution is done along a single direction and for x>0 then using the symmetry it extends to x<0
"""
import time
import numpy as np
import scipy.special as ssp
import scipy.integrate as si
import matplotlib.pyplot as plt


def G(x, m):
    '''
    Analitic solution
    x : float, position
    m : int, energy level
    '''
    return (1/(np.pi)**(1/4))*(1/np.sqrt((2**m)*ssp.gamma(m+1)))*ssp.eval_hermite(m, x)*np.exp(-(x**2)/2)

def V(x, g):
    '''
    potential
    x : position
    '''
    return 1/2 * x**2 + g*x**4

def F(x, y, E, g) :
    
    psi, dpsi = y
    dydt = np.array([dpsi , 2*(V(x, g) - E)*psi ])
    
    return dydt
    

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
        
    sol = si.odeint(F, y, x, args=(E1, *args), tfirst=True)
    psi_tail_1 = sol[-1, 0] # keep the last vaule of psi
    # this value can be very wrong, beacuse we don't know the true
    # value of E, but the import thing is the sign.
    
    # second bound
    while True:
        E2 = E1 + step
        sol = si.odeint(F, y, x, args=(E2, *args), tfirst=True)
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
                
        sol = si.odeint(F, y, x, args=(Em, *args), tfirst=True)
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

N     = 100000                                 # Number of point
x_max = 10                                     # half size of box
steps = 0.005                                  # step for searching of second bound
tau   = 1.0e-13                                # tollerance for bisection
x     = np.linspace(0, x_max, N)               # array of half box, for solving eq in shooting
z     = np.linspace(-x_max, x_max, 2*N + 1)    # total array of distance in the box
g     = 0                                      # coupling of quartic term
n     = 5                                      # label of energy level
i_eig = n                                      # initial eig for eigvalue, theoretical: n + 1/2 for g = 0

#================================================================================
# Computation
#================================================================================

start = time.time()

eigv, psi = shooting(x, n, i_eig ,steps, tau, F, args=(g,))
psi = true_psi(psi, z)

end = time.time() - start

print(f"E_{n} = {eigv:.10f}")
print(f"Elapsed time: {end:.3f}")

#================================================================================
# Plot
#================================================================================

plt.figure(1)
plt.title(f"$\psi(x)$ harmonic osccilator n={n}", fontsize=20)
plt.plot(z, psi + eigv, 'b.', label='$\psi(x)$ numerical' )

plt.plot(z, np.ones(len(z))*eigv, color='black', linestyle='--', label='$E_{%d}=%f$' %(n, eigv))
if g == 0 : plt.plot(z, G(z, n)+eigv, color='black', label='$\psi(x)$ analytics')
plt.plot(z, V(z, g), color='red', label=fr'V(x)= $ 0.5x^2$+ {g}$x^4$')
plt.ylim(eigv - 1, eigv + 1)
plt.grid()
plt.legend(loc='best')
plt.ylabel('$\psi(x)$', fontsize=15)
plt.xlabel('x', fontsize=15)

if g == 0:
    plt.figure(2)
    plt.title("Difference between psi", fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('$\psi(x)_{num}-\psi(x)_{es}$', fontsize=15)
    plt.grid()
    plt.errorbar(z, abs(psi) - abs(G(z, n)), fmt='.')
plt.show()
