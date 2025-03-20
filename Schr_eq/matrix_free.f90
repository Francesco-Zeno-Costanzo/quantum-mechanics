program matrix_free_diagnalizzation

    ! Code for diagonalizing a Hamiltonian of the form H = P^2 + V(x).
    ! The code uses the power method using the conjugate gradient as a method for inverting a matrix.
    ! Everything is implemented in the matrixless form, i.e. no matrices are allocated
    ! instead whenever necessary the result of the matrix-vector product is calculated.
    ! This code is the translation of matrix_free.py

    real*8 , dimension(:), allocatable :: xp            ! Grid point
    real*8 , dimension(:), allocatable :: eigval        ! To store eigenvalues
    real*8 , dimension(:, :), allocatable :: eigvec     ! To store eigenvectors
    real*8 :: tol, h, xl, xr
    common h

    open(unit=10, file='mf_val.dat', status="unknown")  ! File for eigenvalues
    open(unit=20, file='mf_vec.dat', status="unknown")  ! File for eigenvectors

    call cpu_time(start)

    N   = 10000                               ! Number of grid points
    k   = 5                                   ! Number of eigen(values/vectors) to compute
    xl  = -10.d0                              ! Lower limit on x axis
    xr  = 10.d0                               ! Upper limit on x axis
    h   = (xr - xl) / ((N-1) * 1.d0)          ! Grid spacing
    tol = 1.d-10

    allocate(xp(N), eigval(N), eigvec(N, k))

    do i = 1, N
        xp(i) = xl + h * ((i-1)*1.d0)         ! Array of positions
    enddo

    call eig_matrixless(N, k, tol, eigval, eigvec, xp)

    ! Save on file
    print '("      Theoretical                 Computed                      error")'
    do i = 1, k
        write(10, *) eigval(i)
        write(20, *) eigvec(:, i)/dsqrt(h)
        print *, (i-1) + 0.5d0, eigval(i), eigval(i) - ((i-1) + 0.5d0)
    enddo

    write(20, *) xp

    deallocate(xp, eigval, eigvec)
    close(10)
    close(20)

    call cpu_time(finish)
    print '("Elapsed time = ", f16.8," secondi.")', finish-start

end program


subroutine conj_grad_matrixless(N, b, x, xp, tol)

    !===============================================================================
    ! Matrix-free implementation of conjugate gradient method for solving M x = b.
    ! The function that implements matrix vector multiplication
    ! is the H_psi subroutine defined below.
    !
    ! Parameters
    ! ----------
    ! N : int 
    !     size of the input and output vector
    ! b : 1darray
    !     Ordinate or "dependent variable" values.
    ! x : 1darray
    !     Solution of the sistem, i.e. the output of the subroutine
    ! xp : 1darray
    !     grid of our discretized sistem
    ! tol : float, optional
    !     Required tolerance
    !===============================================================================
    
    real*8, dimension(N) :: b, x, xp, r, p, Hpsi
    real*8 alpha, beta, r2, r2_new, tol

    call H_psi(N, x, r, xp)
    
    r  = b - r                ! Residual
    p  = r                    ! Descent direction
    r2 = dot_product(r, r)    ! Norm^2 of residuals

    do while (dsqrt(r2) > tol)
        
        call H_psi(N, p, Hpsi, xp)         ! Compute matrix-free product M @ p
        
        alpha = r2 / dot_product(p, Hpsi)  ! Descent step
        
        x = x + alpha * p                  ! Update position
        r = r - alpha * Hpsi               ! Update residuals
        
        r2_new = dot_product(r, r)         ! Norm^2 of new residuals
        beta   = r2_new / r2               ! Compute step for p

        p  = r + beta * p                  ! Update descent direction
        r2 = r2_new                        ! Update residuals norm
       
    enddo
    
    return
end

subroutine eig_matrixless(N, k, tol, eigval, eigvec, xp)

    !===============================================================================
    ! Compute the eigenvalue decomposition of a matrix using inverse iteration 
    ! in a matrix-free manner, avoiding storing the matrix.
    !
    ! Parameters
    ! ----------
    ! N : int 
    !     size of the input and output vector
    ! k : int
    !     Number of eigen(values/vectors) to compute
    ! tol : float, optional
    !     Required tolerance
    ! eigval : 1darray
    !     eigenvalues found, stored in an array
    ! eigvec : 2darray
    !     Corresponding eigenvectors matrix
    ! xp : 1darray
    !     grid of our discretized sistem
    !===============================================================================

    real*8 , dimension(N) :: xp
    real*8 , dimension(N) :: eigval
    real*8 , dimension(N, k) :: eigvec
    real*8 , dimension(N) :: v_p, v_o, v_l
    real*8 :: tol, l_v, l_o, R1, R2, R3, R

    character :: cr  ! Variable for loading
    cr = char(13)    ! is used to override the shell
    
    do i = 1, k

        write(*, '(A1)', advance='no')  cr   ! I write cr needed to clean the shell

        R1 = 1.d0
        R2 = 1.d0
        R3 = 1.d0
        R  = 1.d0

        ! initializzation
        call random_number(v_p)
        v_p = v_p / dsqrt(dot_product(v_p, v_p))
        l_v = 0.d0

        do while ((R > tol).or.(R3 > tol))
            
            l_o = l_v                                       ! Update
            v_o = v_p
            
            call conj_grad_matrixless(N, v_p, v_p, xp, tol) ! Compute new vector

            v_p = v_p / dsqrt(dot_product(v_p, v_p))

            ! Orthogonalization respect all eigenvectors find previously
            do j = 1, i - 1
                v_p = v_p - dot_product(eigvec(:, j), v_p) * eigvec(:, j)
            end do
            v_p = v_p / sqrt(dot_product(v_p, v_p))

            ! Eigenvalue of v_p: A @ v_p = l_v * v_p
            ! Multiplying by the transposed => (A @ v_p) @ v_p.T = l_v
            ! Using v_p @ v_p.T = 1
            call conj_grad_matrixless(N, v_p, v_l, xp, tol)
            l_v = dot_product(v_p, v_l)

            R1 = dsqrt(dot_product(v_p - v_o, v_p - v_o))
            R2 = dsqrt(dot_product(v_o + v_p, v_o + v_p))
            R3 = dabs(l_v - l_o)  !In eigenvalues the convergence is quadratic
            R  = dmin1(R1, R2)
            
        enddo

        eigvec(:, i) = v_p
        eigval(i) = 1.d0 / l_v

        ! Loading section, I write percentage
        write(*,'(f8.1, a2)', advance='NO') i/float(k)*100," %"
        flush(6) ! Clean the line
        
    enddo

    write(*,*) ! To avoid problem in printing

    return
end

subroutine H_psi(N, psi, Hpsi, xp)

    !===============================================================================
    ! This function computes the product M @ v without storing M.
    ! In this case we have a simple tridiagonal matrix of the form:
    !    
    !           [ 2 -1  0  0  0 ]       [ V0 0  0  0  0 ]
    !           [-1  2 -1  0  0 ]       [ 0  V1 0  0  0 ]
    ! 1/(2*h^2) [ 0 -1  2 -1  0 ]   +   [ 0  0  V3 0  0 ]
    !           [ 0  0 -1  2 -1 ]       [ 0  0  0  V4 0 ]
    !           [ 0  0  0 -1  2 ]       [ 0  0  0  0  V5]
    !    
    ! Parameters
    ! ----------
    ! psi : 1darray
    !     vector to multiply to the matrix
    ! Hpsi : 1darray
    !     Result of the multiplication
    ! xp : 1darray
    !     grid of our discretized sistem
    !    
    !===============================================================================

    real*8 , dimension(N) :: xp, psi, Hpsi
    real*8 :: fact, V, h
    common h

    !**************** Kinetic part ****************
    fact = -0.5d0 / h**2

    Hpsi(1) = fact * (-2.0d0 * psi(1) + psi(2))
    do i = 2, N - 1
        Hpsi(i) = fact * (psi(i - 1) - 2.0d0 * psi(i) + psi(i + 1))
    enddo
    Hpsi(N) = fact * (-2.0d0 * psi(N) + psi(N - 1))

    !**************** Potential part ****************
    do i = 1, N
        V = 0.5d0 * xp(i)**2
        Hpsi(i) = Hpsi(i) + V * psi(i)
    enddo

    return
end
