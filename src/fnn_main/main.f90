
program main

    use fnn

    implicit none
    integer(ik) :: Nx, Ny, Ne, Np, i
    integer, allocatable :: seed(:)
    integer :: seed_size
    type(NeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :), p(:), dy(:, :), dp(:), dx(:, :)
    real(rk), allocatable :: FpT_dy(:, :), FxT_dy(:, :), FpT_dy_total(:), F_dx_dp(:, :)

    ! set random seed
    call random_seed(size=seed_size)
    allocate(seed(seed_size))
    seed = 3141592
    call random_seed(put=seed)

    ! create neural network using a batch size of 16
    ! the architecture and parameters are read from file
    Ne = 16
    network = nn_fromfile(Ne, 'wdir/architecture.txt', 'wdir/parameters.bin')
    print *, 'created model'
    Nx = network % get_input_size()
    Ny = network % get_output_size()
    Np = network % get_num_parameters()

    ! allocate tensors
    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))
    allocate(p(Np))
    allocate(dp(Np))
    allocate(dy(Ny, Ne))
    allocate(dx(Nx, Ne))
    allocate(FpT_dy(Np, Ne))
    allocate(FxT_dy(Nx, Ne))
    allocate(FpT_dy_total(Np))
    allocate(F_dx_dp(Ny, Ne))

    ! fill in input tensors with random numbers
    call rand2d(x)
    call rand1d(p)
    call rand1d(dp)
    call rand2d(dy)
    call rand2d(dx)

    ! save input
    print *, 'writing "wdir/p.bin"'
    open(unit=10, file='wdir/p.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) p
    close(10)
    print *, 'writing "wdir/x.bin"'
    open(unit=10, file='wdir/x.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) x
    close(10)
    print *, 'writing "wdir/dp.bin"'
    open(unit=10, file='wdir/dp.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) dp
    close(10)
    print *, 'writing "wdir/dx.bin"'
    open(unit=10, file='wdir/dx.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) dx
    close(10)
    print *, 'writing "wdir/dy.bin"'
    open(unit=10, file='wdir/dy.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) dy
    close(10)

    ! apply forward
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do
    print *, 'writing "wdir/y.bin"'
    open(unit=10, file='wdir/y.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) y
    close(10)

    ! reset parameters and re-apply forward
    call network % set_parameters(p)
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do
    print *, 'writing "wdir/py.bin"'
    open(unit=10, file='wdir/py.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) y
    close(10)

    ! apply adjoint
    FpT_dy_total(:) = 0
    do i = 1, Ne
        call network % apply_adjoint(i, dy(:, i), FpT_dy(:, i), FxT_dy(:, i))
        FpT_dy_total(:) = FpT_dy_total(:) + FpT_dy(:, i)
    end do
    print *, 'writing "wdir/FpT_dy.bin"'
    open(unit=10, file='wdir/FpT_dy.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) FpT_dy_total
    close(10)
    print *, 'writing "wdir/FxT_dy.bin"'
    open(unit=10, file='wdir/FxT_dy.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) FxT_dy
    close(10)

    ! apply tangent linear
    do i = 1, Ne
        call network % apply_tangent_linear(i, dp(:), dx(:, i), F_dx_dp(:, i))
    end do
    print *, 'writing "wdir/F_dx_dp.bin"'
    open(unit=10, file='wdir/F_dx_dp.bin', form='unformatted', access='stream', action='write', status='replace')
    write(10) F_dx_dp
    close(10)

end program main

