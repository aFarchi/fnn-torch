
program main

    use fnn

    implicit none
    integer(ik) :: Nx, Ny, Ne, i
    integer, allocatable :: seed(:)
    integer :: seed_size
    type(NeuralNetwork) :: network

    real(rk), allocatable :: x(:, :), y(:, :)

    ! set random seed
    call random_seed(size=seed_size)
    allocate(seed(seed_size))
    seed = 3141592
    call random_seed(put=seed)

    ! create neural network using a batch size of 16
    ! the architecture and parameters are read from file
    Ne = 16
    network = nn_fromfile(Ne, 'wdir/architecture.bin', 'wdir/parameters.bin')
    Nx = network % get_input_size()
    Ny = network % get_output_size()

    ! allocate tensors
    allocate(x(Nx, Ne))
    allocate(y(Ny, Ne))

    ! fill in input tensors with random numbers
    call rand2d(x)

    ! apply forward
    do i = 1, Ne
        call network % apply_forward(.true., i, x(:, i), y(:, i))
    end do

    ! save output
    open(unit=10, file='wdir/fnn_out.bin', form='unformatted', access='stream', action='write')
    write(10) x
    write(10) y
    close(10)

end program main

