We want solve:

<img src="https://latex.codecogs.com/svg.image?i&space;\frac{\partial}{\partial&space;t}\psi(x,&space;t)&space;=&space;-\frac{\partial^2}{\partial&space;x^2}\psi(x,&space;t)&space;&space;&plus;&space;V(x)\psi(x,&space;t)" title="i \frac{\partial}{\partial t}\psi(x, t) = -\frac{\partial^2}{\partial x^2}\psi(x, t) + V(x)\psi(x, t)" />

As a first option we could say that the same method of diffusion equations can be used:

<img src="https://latex.codecogs.com/svg.image?i&space;\Biggl(&space;\frac{\psi^{n&space;&plus;1}_j&space;-&space;\psi^n_j}{\Delta&space;t}\Biggl)&space;=&space;-&space;\Biggl(&space;\frac{\psi^{n&space;&plus;1}_{j&plus;1}&space;-&space;2&space;\psi^{n&plus;1}_j&space;&plus;&space;\psi^{n&plus;1}_{j-1}}{\Delta&space;x^2}\Biggl)&space;&plus;&space;V_j&space;\psi^{n&space;&plus;1}_{j}" title="i \Biggl( \frac{\psi^{n +1}_j - \psi^n_j}{\Delta t}\Biggl) = - \Biggl( \frac{\psi^{n +1}_{j+1} - 2 \psi^{n+1}_j + \psi^{n+1}_{j-1}}{\Delta x^2}\Biggl) + V_j \psi^{n +1}_{j}" />

Written in his impicit form. But this method does not preserve the square modulus of the wave function.
We also know that if the Hamiltonian does not depend on time we can write:

<img src="https://latex.codecogs.com/svg.image?i&space;\frac{\partial&space;}{\partial&space;t}&space;\psi(x,&space;t)&space;=&space;H&space;\psi(x,&space;t)&space;\hspace{5mm}&space;\Rightarrow&space;\hspace{5mm}&space;\psi(x,&space;t)&space;=&space;\exp(-iHt)\psi(x,&space;0)&space;\simeq&space;(1&space;-&space;iHt)\psi(x,&space;0)" title="i \frac{\partial }{\partial t} \psi(x, t) = H \psi(x, t) \hspace{5mm} \Rightarrow \hspace{5mm} \psi(x, t) = \exp(-iHt)\psi(x, 0) \simeq (1 - iHt)\psi(x, 0)" />

But the use of this approach always has the problem of preserving the square module. However we can say that the backward evolved wave function of t / 2 is equal to the forward evolved initial one of t / 2:

<img src="https://latex.codecogs.com/svg.image?\\&space;\exp(iHt/2)&space;\psi(x,&space;t)&space;=&space;\exp(-iHt/2)\psi(x,&space;0)\\\\\text{and&space;approximating:}\\\\(1&space;&plus;&space;iHt/2)\psi(x,&space;t)&space;=&space;(1&space;-&space;iHt/2)\psi(x,&space;0)&space;\\\\\psi(x,&space;t)&space;=&space;\frac{(1&space;-&space;iHt/2)}{(1&space;&plus;&space;iHt/2)}\psi(x,&space;0)" title="\\ \exp(iHt/2) \psi(x, t) = \exp(-iHt/2)\psi(x, 0)\\\\\text{and approximating:}\\\\(1 + iHt/2)\psi(x, t) = (1 - iHt/2)\psi(x, 0) \\\\\psi(x, t) = \frac{(1 - iHt/2)}{(1 + iHt/2)}\psi(x, 0)" />

In this way the term in front of the initial wave function has the form of a complex number divided by its conjugate, which has modulus equal to one so the square modulus of the wave function is conserved.
The following is an example of the code output:

![](tunnel_barriera.gif)