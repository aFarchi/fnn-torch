
module fnn

    use iso_fortran_env, only: int32, int64, real32, real64, real128

    implicit none

    private
    public :: rk, r0, ik, rand1d, rand2d, NeuralNetwork, nn_fromfile

    integer, parameter :: rk = real64
    integer, parameter :: r0 = real64
    integer, parameter :: ik = int32

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type :: Layer
        private
        integer(ik) :: input_size
        integer(ik) :: output_size
        integer(ik) :: batch_size
        integer(ik) :: num_parameters
        real(rk), allocatable :: parameters(:)
        real(rk), allocatable :: forward_input(:, :)
        real(rk), allocatable :: tangent_linear_input(:, :)
        real(rk), allocatable :: adjoint_input(:, :)
    contains
        procedure, pass :: read_parameters => layer_read_parameters
        procedure, pass :: set_parameters => layer_set_parameters
        procedure, pass :: get_parameters => layer_get_parameters
        procedure, pass :: apply_forward => layer_apply_forward
        procedure, pass :: apply_tangent_linear => layer_apply_tangent_linear
        procedure, pass :: apply_adjoint => layer_apply_adjoint
    end type Layer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type :: LayerContainer
        private
        class(Layer), allocatable :: this_layer
    end type

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type :: NeuralNetwork
        private
        type(LayerContainer) :: layer_container
    contains
        procedure, public, pass :: get_input_size => nn_get_input_size
        procedure, public, pass :: get_output_size => nn_get_output_size
        procedure, public, pass :: get_num_parameters => nn_get_num_parameters
        procedure, public, pass :: set_parameters => nn_set_parameters
        procedure, public, pass :: get_parameters => nn_get_parameters
        procedure, public, pass :: apply_forward => nn_apply_forward
        procedure, public, pass :: apply_tangent_linear => nn_apply_tangent_linear
        procedure, public, pass :: apply_adjoint => nn_apply_adjoint
    end type NeuralNetwork

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: SequentialLayer
        private
        integer(ik) :: num_layers
        type(LayerContainer), allocatable :: list_layers(:)
        integer(ik), allocatable :: ip_start(:)
        integer(ik), allocatable :: ip_end(:)
    contains
        procedure, pass :: read_parameters => sequential_read_parameters
        procedure, pass :: set_parameters => sequential_set_parameters
        procedure, pass :: get_parameters => sequential_get_parameters
        procedure, pass :: apply_forward => sequential_apply_forward
        procedure, pass :: apply_tangent_linear => sequential_apply_tangent_linear
        procedure, pass :: apply_adjoint => sequential_apply_adjoint
    end type SequentialLayer
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: LinearLayer
        private
    contains
        procedure, pass :: apply_forward => linear_apply_forward
        procedure, pass :: apply_tangent_linear => linear_apply_tangent_linear
        procedure, pass :: apply_adjoint => linear_apply_adjoint
    end type LinearLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: SkipConnectionLayer
        private
        type(LayerContainer), allocatable :: layer_container
    contains
        procedure, pass :: read_parameters => skip_connection_read_parameters
        procedure, pass :: set_parameters => skip_connection_set_parameters
        procedure, pass :: get_parameters => skip_connection_get_parameters
        procedure, pass :: apply_forward => skip_connection_apply_forward
        procedure, pass :: apply_tangent_linear => skip_connection_apply_tangent_linear
        procedure, pass :: apply_adjoint => skip_connection_apply_adjoint
    end type SkipConnectionLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: RegionSplitLayer
        private
        type(LayerContainer), allocatable :: layer_sh
        type(LayerContainer), allocatable :: layer_tr
        type(LayerContainer), allocatable :: layer_nh
        real(rk), allocatable :: forward_output_sh(:, :)
        real(rk), allocatable :: forward_output_tr(:, :)
        real(rk), allocatable :: forward_output_nh(:, :)
        real(rk), allocatable :: tl_output(:)
        integer(ik) :: ip_start_sh
        integer(ik) :: ip_end_sh
        integer(ik) :: ip_start_tr
        integer(ik) :: ip_end_tr
        integer(ik) :: ip_start_nh
        integer(ik) :: ip_end_nh
        integer(ik) :: index_sin_lat
        real(rk) :: sin_30
        real(rk) :: sin_20
    contains
        procedure, pass :: read_parameters => region_split_read_parameters
        procedure, pass :: set_parameters => region_split_set_parameters
        procedure, pass :: get_parameters => region_split_get_parameters
        procedure, pass :: apply_forward => region_split_apply_forward
        procedure, pass :: apply_tangent_linear => region_split_apply_tangent_linear
        procedure, pass :: apply_adjoint => region_split_apply_adjoint
    end type RegionSplitLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: NormalisationLayer
        private
    contains
        procedure, pass :: apply_forward => normalisation_apply_forward
        procedure, pass :: apply_tangent_linear => normalisation_apply_tangent_linear
        procedure, pass :: apply_adjoint => normalisation_apply_adjoint
    end type NormalisationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: AppendStaticInputLayer
        private
    contains
        procedure, pass :: apply_forward => append_static_input_apply_forward
        procedure, pass :: apply_tangent_linear => append_static_input_apply_tangent_linear
        procedure, pass :: apply_adjoint => append_static_input_apply_adjoint
    end type AppendStaticInputLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(Layer) :: ActivationLayer
        private
        real(rk), allocatable, public :: x_prime(:, :)
    contains
        procedure, pass :: apply_tangent_linear => activation_apply_tangent_linear
        procedure, pass :: apply_adjoint => activation_apply_adjoint
    end type ActivationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(ActivationLayer) :: ReluActivationLayer
        private
    contains
        procedure, pass :: apply_forward => relu_activation_apply_forward
    end type ReluActivationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(ActivationLayer) :: GeluActivationLayer
        private
    contains
        procedure, pass :: apply_forward => gelu_activation_apply_forward
    end type GeluActivationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(ActivationLayer) :: TanhActivationLayer
        private
    contains
        procedure, pass :: apply_forward => tanh_activation_apply_forward
    end type TanhActivationLayer

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type, extends(ActivationLayer) :: DropoutLayer
        private
        real(rk) :: dropout_rate
    contains
        procedure, pass :: apply_forward => dropout_apply_forward
    end type DropoutLayer

contains

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine rand1d(x)
        real(rk), intent(out) :: x(:)
        call random_number(x)
    end subroutine rand1d

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine rand2d(x)
        real(rk), intent(out) :: x(:, :)
        call random_number(x)
    end subroutine rand2d

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    logical function is_frozen(fileunit)
        integer(ik), intent(in) :: fileunit
        character(len=100) :: frozen_text
        read(fileunit, fmt=*) frozen_text
        select case(trim(frozen_text))
            case('frozen')
                is_frozen = .true.
            case default
                is_frozen = .false.
            end select
    end function is_frozen

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class NeuralNetwork
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer(ik) function nn_get_input_size(self) result(input_size)
        class(NeuralNetwork), intent(in) :: self
        input_size = self % layer_container % this_layer % input_size
    end function nn_get_input_size

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer(ik) function nn_get_output_size(self) result(output_size)
        class(NeuralNetwork), intent(in) :: self
        output_size = self % layer_container % this_layer % output_size
    end function nn_get_output_size

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer(ik) function nn_get_num_parameters(self) result(num_parameters)
        class(NeuralNetwork), intent(in) :: self
        num_parameters = self % layer_container % this_layer % num_parameters
    end function nn_get_num_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_set_parameters(self, new_parameters)
        class(NeuralNetwork), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        call self % layer_container % this_layer % set_parameters(new_parameters)
    end subroutine nn_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_get_parameters(self, parameters)
        class(NeuralNetwork), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        call self % layer_container % this_layer % get_parameters(parameters)
    end subroutine nn_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_apply_forward(self, train, member, x, y)
        class(NeuralNetwork), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        call self % layer_container % this_layer % apply_forward(train, member, x, y)
    end subroutine nn_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_apply_tangent_linear(self, member, dp, dx, dy)
        class(NeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        call self % layer_container % this_layer % apply_tangent_linear(member, dp, dx, dy)
    end subroutine nn_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine nn_apply_adjoint(self, member, dy, dp, dx)
        class(NeuralNetwork), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        call self % layer_container % this_layer % apply_adjoint(member, dy, dp, dx)
    end subroutine nn_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class Layer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_read_parameters(self, fileunit)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: fileunit
        real(r0), allocatable :: the_parameters(:)
        ! read in r0 precision
        allocate(the_parameters(size(self % parameters)))
        read(fileunit) the_parameters
        ! cast to rk precision
        self % parameters = the_parameters
    end subroutine layer_read_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_set_parameters(self, new_parameters)
        class(Layer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        if ( self % num_parameters > 0 ) then
            self % parameters = new_parameters
        end if
    end subroutine layer_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_get_parameters(self, parameters)
        class(Layer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        parameters = self % parameters
    end subroutine layer_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_apply_forward(self, train, member, x, y)
        class(Layer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        print *, 'WARNING: using non-implemented method Layer::apply_forward()'
    end subroutine layer_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_apply_tangent_linear(self, member, dp, dx, dy)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        print *, 'WARNING: using non-implemented method Layer::apply_tangent_linear()'
    end subroutine layer_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine layer_apply_adjoint(self, member, dy, dp, dx)
        class(Layer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        print *, 'WARNING: using non-implemented method Layer::apply_adjoint()'
    end subroutine layer_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class SequentialLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_read_parameters(self, fileunit)
        class(SequentialLayer), intent(inout) :: self
        integer(ik), intent(in) :: fileunit
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % read_parameters(fileunit)
        end do
    end subroutine sequential_read_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_set_parameters(self, new_parameters)
        class(SequentialLayer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % set_parameters(&
                new_parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine sequential_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_get_parameters(self, parameters)
        class(SequentialLayer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        integer(ik) :: i
        do i = 1, self % num_layers
            call self % list_layers(i) % this_layer % get_parameters(&
                parameters(self % ip_start(i):self % ip_end(i)))
        end do
    end subroutine sequential_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_apply_forward(self, train, member, x, y)
        class(SequentialLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_forward(&
                train, member, x, y)
        else
            call self % list_layers(1) % this_layer % apply_forward(&
                train, member, x,&
                self % list_layers(2) % this_layer % forward_input(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_forward(&
                    train, member,&
                    self % list_layers(i) % this_layer % forward_input(:, member),&
                    self % list_layers(i+1) % this_layer % forward_input(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_forward(&
                train, member,&
                self % list_layers(self % num_layers) % this_layer % forward_input(:, member), y)
        end if
    end subroutine sequential_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_apply_tangent_linear(self, member, dp, dx, dy)
        class(SequentialLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        integer(ik) :: i
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_tangent_linear(member, dp, dx, dy)
        else
            call self % list_layers(1) % this_layer % apply_tangent_linear(member,&
                dp(self % ip_start(1):self % ip_end(1)), dx,&
                self % list_layers(2) % this_layer % tangent_linear_input(:, member))
            do i = 2, self % num_layers - 1
                call self % list_layers(i) % this_layer % apply_tangent_linear(member,&
                    dp(self % ip_start(i):self % ip_end(i)),&
                    self % list_layers(i) % this_layer % tangent_linear_input(:, member),&
                    self % list_layers(i+1) % this_layer % tangent_linear_input(:, member))
            end do
            call self % list_layers(self % num_layers) % this_layer % apply_tangent_linear(member,&
                dp(self % ip_start(self % num_layers):self % ip_end(self % num_layers)),&
                self % list_layers(self % num_layers) % this_layer % tangent_linear_input(:, member), dy)
        end if
    end subroutine sequential_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine sequential_apply_adjoint(self, member, dy, dp, dx)
        class(sequentialLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        integer(ik) :: i
        if ( self % num_layers == 1 ) then
            call self % list_layers(1) % this_layer % apply_adjoint(member, dy, dp, dx)
        else
            call self % list_layers(self % num_layers) % this_layer % apply_adjoint(member, dy,&
                dp(self % ip_start(self % num_layers):self % ip_end(self % num_layers)),&
                self % list_layers(self % num_layers-1) % this_layer % adjoint_input(:, member))
            do i = self % num_layers - 1, 2, -1
                call self % list_layers(i) % this_layer % apply_adjoint(member,&
                    self % list_layers(i) % this_layer % adjoint_input(:, member),&
                    dp(self % ip_start(i):self % ip_end(i)),&
                    self % list_layers(i-1) % this_layer % adjoint_input(:, member))
            end do
            call self % list_layers(1) % this_layer % apply_adjoint(member,&
                self % list_layers(1) % this_layer % adjoint_input(:, member),&
                dp(self % ip_start(1):self % ip_end(1)), dx)
        end if
    end subroutine sequential_apply_adjoint
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class LinearLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine linear_apply_forward(self, train, member, x, y)
        class(LinearLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        self % forward_input(:, member) = x
        y = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            x)
        y = y + self % parameters(1:self % output_size)
    end subroutine linear_apply_forward
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine linear_apply_tangent_linear(self, member, dp, dx, dy)
        class(LinearLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = matmul(&
            reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size]),&
            dx)
        if ( self % num_parameters > 0 ) then
            dy = dy + matmul(&
                reshape(dp(self % output_size+1:self % output_size*(self % input_size+1)),&
                [self % output_size, self % input_size]),&
                self % forward_input(:, member))
            dy = dy + dp(1:self % output_size)
        end if
    end subroutine linear_apply_tangent_linear
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine linear_apply_adjoint(self, member, dy, dp, dx)
        class(LinearLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        if ( self % num_parameters > 0 ) then
            dp(1:self % output_size) = dy
            dp(self % output_size+1:self % output_size*(self % input_size+1)) = reshape(matmul(&
                reshape(dp(1:self % output_size), [self % output_size, 1]),&
                reshape(self % forward_input(:, member), [1, self % input_size])),&
                [self % output_size*self % input_size])
        end if
        dx = matmul(transpose(reshape(self % parameters(self % output_size+1:self % output_size*(self % input_size+1)),&
            [self % output_size, self % input_size])), dy)
    end subroutine linear_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class SkipConnectionLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine skip_connection_read_parameters(self, fileunit)
        class(SkipConnectionLayer), intent(inout) :: self
        integer(ik), intent(in) :: fileunit
        call self % layer_container % this_layer % read_parameters(fileunit)
    end subroutine skip_connection_read_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine skip_connection_set_parameters(self, new_parameters)
        class(SkipConnectionLayer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        call self % layer_container % this_layer % set_parameters(new_parameters)
    end subroutine skip_connection_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine skip_connection_get_parameters(self, parameters)
        class(SkipConnectionLayer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        call self % layer_container % this_layer % get_parameters(parameters)
    end subroutine skip_connection_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine skip_connection_apply_forward(self, train, member, x, y)
        class(SkipConnectionLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y(1:self % input_size) = x
        call self % layer_container % this_layer % apply_forward(train, member, x,&
                y(self % input_size+1:self % output_size))
    end subroutine skip_connection_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine skip_connection_apply_tangent_linear(self, member, dp, dx, dy)
        class(SkipConnectionLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy(1:self % input_size) = dx
        call self % layer_container % this_layer % apply_tangent_linear(member, dp, dx,&
                dy(self % input_size+1:self % output_size))
    end subroutine skip_connection_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine skip_connection_apply_adjoint(self, member, dy, dp, dx)
        class(SkipConnectionLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        call self % layer_container % this_layer % apply_adjoint(member,&
                dy(self % input_size+1:self % output_size), dp, dx)
        dx = dx + dy(1:self % input_size)
    end subroutine skip_connection_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class RegionSplitLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine region_split_read_parameters(self, fileunit)
        class(RegionSplitLayer), intent(inout) :: self
        integer(ik), intent(in) :: fileunit
        call self % layer_sh % this_layer % read_parameters(fileunit)
        call self % layer_tr % this_layer % read_parameters(fileunit)
        call self % layer_nh % this_layer % read_parameters(fileunit)
    end subroutine region_split_read_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine region_split_set_parameters(self, new_parameters)
        class(RegionSplitLayer), intent(inout) :: self
        real(rk), intent(in) :: new_parameters(:)
        call self % layer_sh % this_layer % set_parameters(&
            new_parameters(self % ip_start_sh:self % ip_end_sh))
        call self % layer_tr % this_layer % set_parameters(&
            new_parameters(self % ip_start_tr:self % ip_end_tr))
        call self % layer_nh % this_layer % set_parameters(&
            new_parameters(self % ip_start_nh:self % ip_end_nh))
    end subroutine region_split_set_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine region_split_get_parameters(self, parameters)
        class(RegionSplitLayer), intent(in) :: self
        real(rk), intent(out) :: parameters(:)
        call self % layer_sh % this_layer % get_parameters(&
            parameters(self % ip_start_sh:self % ip_end_sh))
        call self % layer_tr % this_layer % get_parameters(&
            parameters(self % ip_start_tr:self % ip_end_tr))
        call self % layer_nh % this_layer % get_parameters(&
            parameters(self % ip_start_nh:self % ip_end_nh))
    end subroutine region_split_get_parameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine region_split_apply_forward(self, train, member, x, y)
        class(RegionSplitLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        real(rk) :: sin_x, alpha_nh, alpha_sh, alpha_tr
        self % forward_input(:, member) = x
        sin_x = x(self % index_sin_lat)
        alpha_nh = (sin_x - self % sin_20) / (self % sin_30 - self % sin_20)
        alpha_nh = max(0.0_rk, min(1.0_rk, alpha_nh))
        alpha_sh = -(sin_x + self % sin_20) / (self % sin_30 - self % sin_20)
        alpha_sh = max(0.0_rk, min(1.0_rk, alpha_sh))
        alpha_tr = (1.0_rk - alpha_nh) * (1.0_rk - alpha_sh)
        call self % layer_sh % this_layer % apply_forward(&
            train, member, x,&
            self % forward_output_sh(member, :))
        y = self % forward_output_sh(member, :) * alpha_sh
        call self % layer_tr % this_layer % apply_forward(&
            train, member, x,&
            self % forward_output_tr(member, :))
        y = y + self % forward_output_tr(member, :) * alpha_tr
        call self % layer_nh % this_layer % apply_forward(&
            train, member, x,&
            self % forward_output_nh(member, :))
        y = y + self % forward_output_nh(member, :) * alpha_nh
    end subroutine region_split_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine region_split_apply_tangent_linear(self, member, dp, dx, dy)
        class(RegionSplitLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        real(rk) :: sin_x, alpha_nh, alpha_sh, alpha_tr
        sin_x = self % forward_input(self % index_sin_lat, member)
        alpha_nh = (sin_x - self % sin_20) / (self % sin_30 - self % sin_20)
        alpha_nh = max(0.0_rk, min(1.0_rk, alpha_nh))
        alpha_sh = -(sin_x + self % sin_20) / (self % sin_30 - self % sin_20)
        alpha_sh = max(0.0_rk, min(1.0_rk, alpha_sh))
        alpha_tr = (1.0_rk - alpha_nh) * (1.0_rk - alpha_sh)
        call self % layer_sh % this_layer % apply_tangent_linear(&
            member, dp(self % ip_start_sh:self % ip_end_sh), dx,&
            self % tl_output)
        dy = self % tl_output * alpha_sh
        call self % layer_tr % this_layer % apply_tangent_linear(&
            member, dp(self % ip_start_tr:self % ip_end_tr), dx,&
            self % tl_output)
        dy = dy + self % tl_output * alpha_tr
        call self % layer_nh % this_layer % apply_tangent_linear(&
            member, dp(self % ip_start_nh:self % ip_end_nh), dx,&
            self % tl_output)
        dy = dy + self % tl_output * alpha_nh
        if ( - self % sin_30 < sin_x .and. sin_x < - self % sin_20 ) then
            dy = dy - dx(self % index_sin_lat) * self % forward_output_sh(member, :) / (self % sin_30 - self % sin_20)
            dy = dy + dx(self % index_sin_lat) * self % forward_output_tr(member, :) / (self % sin_30 - self % sin_20)
        end if
        if ( self % sin_20 < sin_x .and. sin_x < self % sin_30 ) then
            dy = dy - dx(self % index_sin_lat) * self % forward_output_tr(member, :) / (self % sin_30 - self % sin_20)
            dy = dy + dx(self % index_sin_lat) * self % forward_output_nh(member, :) / (self % sin_30 - self % sin_20)
        end if
    end subroutine region_split_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine region_split_apply_adjoint(self, member, dy, dp, dx)
        class(RegionSplitLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        real(rk) :: sin_x, alpha_nh, alpha_sh, alpha_tr
        sin_x = self % forward_input(self % index_sin_lat, member)
        alpha_nh = (sin_x - self % sin_20) / (self % sin_30 - self % sin_20)
        alpha_nh = max(0.0_rk, min(1.0_rk, alpha_nh))
        alpha_sh = -(sin_x + self % sin_20) / (self % sin_30 - self % sin_20)
        alpha_sh = max(0.0_rk, min(1.0_rk, alpha_sh))
        alpha_tr = (1.0_rk - alpha_nh) * (1.0_rk - alpha_sh)
        call self % layer_sh % this_layer % apply_adjoint(&
            member, dy,&
            dp(self % ip_start_sh:self % ip_end_sh),&
            self % tl_output)
        dp(self % ip_start_sh:self % ip_end_sh) = alpha_sh * dp(self % ip_start_sh:self % ip_end_sh)
        dx = alpha_sh * self % tl_output
        call self % layer_tr % this_layer % apply_adjoint(&
            member, dy,&
            dp(self % ip_start_tr:self % ip_end_tr),&
            self % tl_output)
        dp(self % ip_start_tr:self % ip_end_tr) = alpha_tr * dp(self % ip_start_tr:self % ip_end_tr)
        dx = dx + alpha_tr * self % tl_output
        call self % layer_nh % this_layer % apply_adjoint(&
            member, dy,&
            dp(self % ip_start_nh:self % ip_end_nh),&
            self % tl_output)
        dp(self % ip_start_nh:self % ip_end_nh) = alpha_nh * dp(self % ip_start_nh:self % ip_end_nh)
        dx = dx + alpha_nh * self % tl_output
        if ( - self % sin_30 < sin_x .and. sin_x < - self % sin_20 ) then
            dx(self % index_sin_lat) = dx(self % index_sin_lat) - dot_product(self % forward_output_sh(member, :), dy) / (self % sin_30 - self % sin_20)
            dx(self % index_sin_lat) = dx(self % index_sin_lat) + dot_product(self % forward_output_tr(member, :), dy) / (self % sin_30 - self % sin_20)
        end if
        if ( self % sin_20 < sin_x .and. sin_x < self % sin_30 ) then
            dx(self % index_sin_lat) = dx(self % index_sin_lat) - dot_product(self % forward_output_tr(member, :), dy) / (self % sin_30 - self % sin_20)
            dx(self % index_sin_lat) = dx(self % index_sin_lat) + dot_product(self % forward_output_nh(member, :), dy) / (self % sin_30 - self % sin_20)
        end if
    end subroutine region_split_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class NormalisationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine normalisation_apply_forward(self, train, member, x, y)
        class(NormalisationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        self % forward_input(:, member) = x
        y = self % parameters(1:self % input_size) * x&
            + self % parameters(self % input_size+1:2*self % input_size)
    end subroutine normalisation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine normalisation_apply_tangent_linear(self, member, dp, dx, dy)
        class(NormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % parameters(1:self % input_size) * dx
        if ( self % num_parameters > 0 ) then
             dy = dy + dp(1:self % input_size) * self % forward_input(:, member)&
                  + dp(self % input_size+1:2*self % input_size)
        end if
    end subroutine normalisation_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine normalisation_apply_adjoint(self, member, dy, dp, dx)
        class(NormalisationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % parameters(1:self % input_size) * dy
        if ( self % num_parameters > 0 ) then
            dp(1:self % input_size) = self % forward_input(:, member) * dy
            dp(self % input_size+1:2*self % input_size) = dy
        end if
    end subroutine normalisation_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class AppendStaticInputLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine append_static_input_apply_forward(self, train, member, x, y)
        class(AppendStaticInputLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y(1:self % input_size) = x
        y(self % input_size+1:self % output_size) = self % parameters
    end subroutine append_static_input_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine append_static_input_apply_tangent_linear(self, member, dp, dx, dy)
        class(AppendStaticInputLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy(1:self % input_size) = dx
        if ( self % num_parameters > 0 ) then
            dy(self % input_size+1:self % output_size) = dp
        else
            dy(self % input_size+1:self % output_size) = 0
        end if
    end subroutine append_static_input_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine append_static_input_apply_adjoint(self, member, dy, dp, dx)
        class(AppendStaticInputLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = dy(1:self % input_size)
        if ( self % num_parameters > 0 ) then
            dp = dy(self % input_size+1:self % output_size)
        end if
    end subroutine append_static_input_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class ActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine activation_apply_tangent_linear(self, member, dp, dx, dy)
        class(ActivationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: dp(:)
        real(rk), intent(in) :: dx(:)
        real(rk), intent(out) :: dy(:)
        dy = self % x_prime(:, member) * dx
    end subroutine activation_apply_tangent_linear

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine activation_apply_adjoint(self, member, dy, dp, dx)
        class(ActivationLayer), intent(inout) :: self
        integer(ik), intent(in) :: member
        real(rk), intent(inout) :: dy(:)
        real(rk), intent(out) :: dp(:)
        real(rk), intent(out) :: dx(:)
        dx = self % x_prime(:, member) * dy
    end subroutine activation_apply_adjoint

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class ReluActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine relu_activation_apply_forward(self, train, member, x, y)
        class(ReluActivationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        do i = 1, size(y)
            if (x(i) > 0) then
                y(i) = x(i)
                self % x_prime(i, member) = 1
            else
                y(i) = 0
                self % x_prime(i, member) = 0
            end if
        end do
    end subroutine relu_activation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class GeluActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine gelu_activation_apply_forward(self, train, member, x, y)
        class(GeluActivationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        real(rk), parameter :: inv_sqrt2 = 1.0 / sqrt(2.0)
        real(rk), parameter :: inv_sqrt2pi = 1.0 / sqrt(2.0 * acos(-1.0))
        do i = 1, size(y)
            y(i) = 0.5 * x(i) * (1.0 + erf(x(i) * inv_sqrt2))
            self % x_prime(i, member) = 0.5 * (1.0 + erf(x(i) * inv_sqrt2)) +&
                    x(i) * inv_sqrt2pi * exp(-0.5 * x(i) * x(i))
        end do
    end subroutine gelu_activation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class TanhActivationLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine tanh_activation_apply_forward(self, train, member, x, y)
        class(TanhActivationLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        y = tanh(x)
        self % x_prime(:, member) = 1 - y**2
    end subroutine tanh_activation_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! implementation of class DropoutLayer
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine dropout_apply_forward(self, train, member, x, y)
        class(DropoutLayer), intent(inout) :: self
        logical, intent(in) :: train
        integer(ik), intent(in) :: member
        real(rk), intent(in) :: x(:)
        real(rk), intent(out) :: y(:)
        integer(ik) :: i
        if (train) then
            call rand1d(self % x_prime(:, member))
            do i = 1, self % input_size
                if ( self % x_prime(i, member) < self % dropout_rate ) then
                    self % x_prime(i, member) = 0
                else
                    self % x_prime(i, member) = 1 / (1 - self % dropout_rate)
                endif
            enddo
        else
            self % x_prime(:, member) = 1
        end if
        y = self % x_prime(:, member) * x
    end subroutine dropout_apply_forward

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! constructors
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(NeuralNetwork) function nn_fromfile(batch_size, filename_txt, filename_bin) result(self)
        integer(ik), intent(in) :: batch_size
        character(len=*), intent(in) :: filename_txt
        character(len=*), intent(in) :: filename_bin
        integer(ik) :: fileunit
        ! read architecture
        open(newunit=fileunit, file=filename_txt, action='read')
        self % layer_container = layer_container_fromfile(batch_size, fileunit)
        close(fileunit)
        ! read parameters
        open(newunit=fileunit, file=filename_bin, form='unformatted', access='stream', action='read')
        call self % layer_container % this_layer % read_parameters(fileunit)
        close(fileunit)
    end function nn_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(LayerContainer) function layer_container_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        character(len=100) :: layer_name
        read(fileunit, fmt=*) layer_name
        select case(trim(layer_name)) ! loop over all layers here
            case('linear')
                allocate(LinearLayer::self % this_layer)
                self % this_layer = linear_layer_fromfile(batch_size, fileunit)
            case('skip_connection')
                allocate(SkipConnectionLayer::self % this_layer)
                self % this_layer = skip_connection_layer_fromfile(batch_size, fileunit)
            case('normalisation')
                allocate(NormalisationLayer::self % this_layer)
                self % this_layer = normalisation_layer_fromfile(batch_size, fileunit)
            case('append_static_input')
                allocate(AppendStaticInputLayer::self % this_layer)
                self % this_layer = append_static_input_layer_fromfile(batch_size, fileunit)
            case('relu_activation')
                allocate(ReluActivationLayer::self % this_layer)
                self % this_layer = relu_activation_layer_fromfile(batch_size, fileunit)
            case('gelu_activation')
                allocate(GeluActivationLayer::self % this_layer)
                self % this_layer = gelu_activation_layer_fromfile(batch_size, fileunit)
            case('tanh_activation')
                allocate(TanhActivationLayer::self % this_layer)
                self % this_layer = tanh_activation_layer_fromfile(batch_size, fileunit)
            case('dropout')
                allocate(DropoutLayer::self % this_layer)
                self % this_layer = dropout_layer_fromfile(batch_size, fileunit)
            case('region_split')
                allocate(RegionSplitLayer::self % this_layer)
                self % this_layer = region_split_layer_fromfile(batch_size, fileunit)
            case default
                allocate(SequentialLayer::self % this_layer)
                self % this_layer = sequential_layer_fromfile(batch_size, fileunit)
        end select
    end function layer_container_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(Layer) function construct_layer(input_size, output_size, batch_size, num_parameters, frozen) result (self)
        integer(ik), intent(in) :: input_size
        integer(ik), intent(in) :: output_size
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: num_parameters
        logical, intent(in) :: frozen
        self % input_size = input_size
        self % output_size = output_size
        self % batch_size = batch_size
        self % num_parameters = num_parameters
        allocate(self % parameters(self % num_parameters))
        allocate(self % forward_input(self % input_size, self % batch_size))
        allocate(self % tangent_linear_input(self % input_size, self % batch_size))
        allocate(self % adjoint_input(self % output_size, self % batch_size))
        self % parameters = 0
        self % forward_input = 0
        self % tangent_linear_input = 0
        self % adjoint_input = 0
        if ( frozen ) then
            self % num_parameters = 0
        end if
    end function construct_layer
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(SequentialLayer) function sequential_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        integer(ik) :: i
        integer(ik) :: ip
        read(fileunit, *) self % num_layers
        allocate(self % list_layers(self % num_layers))
        allocate(self % ip_start(self % num_layers))
        allocate(self % ip_end(self % num_layers))
        ip = 0
        do i = 1, self % num_layers
            self % list_layers(i) = layer_container_fromfile(batch_size, fileunit)
            self % ip_start(i) = ip + 1
            ip = ip + self % list_layers(i) % this_layer % num_parameters
            self % ip_end(i) = ip
        end do
        self % Layer = construct_layer(&
            self % list_layers(1) % this_layer % input_size,&
            self % list_layers(self % num_layers) % this_layer % output_size,&
            batch_size,&
            0,&
            .false.&
        )
        self % num_parameters = ip
    end function sequential_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(LinearLayer) function linear_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        logical :: frozen
        integer(ik) :: input_size
        integer(ik) :: output_size
        integer(ik) :: num_parameters
        frozen = is_frozen(fileunit)
        read(fileunit, *) input_size
        read(fileunit, *) output_size
        num_parameters = (input_size+1) * output_size
        self % Layer = construct_layer(input_size, output_size, batch_size, num_parameters, frozen)
    end function linear_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(SkipConnectionLayer) function skip_connection_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % layer_container = layer_container_fromfile(batch_size, fileunit)
        self % Layer = construct_layer(&
            self % layer_container % this_layer % input_size,&
            self % layer_container % this_layer % input_size + self % layer_container % this_layer % output_size,&
            batch_size,&
            0,&
            .false.&
        )
        self % num_parameters = self % layer_container % this_layer % num_parameters
    end function skip_connection_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(RegionSplitLayer) function region_split_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        read(fileunit, *) self % index_sin_lat
        self % layer_sh = layer_container_fromfile(batch_size, fileunit)
        self % layer_tr = layer_container_fromfile(batch_size, fileunit)
        self % layer_nh = layer_container_fromfile(batch_size, fileunit)
        self % Layer = construct_layer(&
            self % layer_sh % this_layer % input_size,&
            self % layer_sh % this_layer % output_size,&
            batch_size,&
            0,&
            .false.&
        )
        self % num_parameters = self % layer_sh % this_layer % num_parameters
        self % num_parameters = self % num_parameters + self % layer_tr % this_layer % num_parameters
        self % num_parameters = self % num_parameters + self % layer_nh % this_layer % num_parameters
        self % ip_start_sh = 1
        self % ip_end_sh = self % ip_start_sh + self % layer_sh % this_layer % num_parameters - 1
        self % ip_start_tr = self % ip_end_sh + 1
        self % ip_end_tr = self % ip_start_tr + self % layer_tr % this_layer % num_parameters - 1
        self % ip_start_nh = self % ip_end_tr + 1
        self % ip_end_nh = self % ip_start_nh + self % layer_nh % this_layer % num_parameters - 1
        self % sin_20 = sin(20.0_rk * acos(-1.0) / 180.0_rk)
        self % sin_30 = sin(30.0_rk * acos(-1.0) / 180.0_rk)
        allocate(self % forward_output_sh(batch_size, self % layer_sh % this_layer % output_size))
        allocate(self % forward_output_tr(batch_size, self % layer_tr % this_layer % output_size))
        allocate(self % forward_output_nh(batch_size, self % layer_nh % this_layer % output_size))
        allocate(self % tl_output(self % layer_sh % this_layer % output_size))
    end function region_split_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(NormalisationLayer) function normalisation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        logical :: frozen
        integer(ik) :: input_size
        frozen = is_frozen(fileunit)
        read(fileunit, *) input_size
        self % Layer = construct_layer(input_size, input_size, batch_size, 2 * input_size, frozen)
    end function normalisation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(AppendStaticInputLayer) function append_static_input_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        logical :: frozen
        integer(ik) :: input_size
        integer(ik) :: num_parameters
        frozen = is_frozen(fileunit)
        read(fileunit, *) input_size
        read(fileunit, *) num_parameters
        self % Layer = construct_layer(input_size, input_size+num_parameters, batch_size, num_parameters, frozen)
    end function append_static_input_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(ActivationLayer) function activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        integer(ik) :: input_size
        read(fileunit, *) input_size
        self % Layer = construct_layer(input_size, input_size, batch_size, 0, .false.)
        allocate(self % x_prime(input_size, batch_size))
    end function activation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(ReluActivationLayer) function relu_activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % ActivationLayer = activation_layer_fromfile(batch_size, fileunit)
    end function relu_activation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(GeluActivationLayer) function gelu_activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % ActivationLayer = activation_layer_fromfile(batch_size, fileunit)
    end function gelu_activation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(TanhActivationLayer) function tanh_activation_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % ActivationLayer = activation_layer_fromfile(batch_size, fileunit)
    end function tanh_activation_layer_fromfile

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    type(DropoutLayer) function dropout_layer_fromfile(batch_size, fileunit) result (self)
        integer(ik), intent(in) :: batch_size
        integer(ik), intent(in) :: fileunit
        self % ActivationLayer = activation_layer_fromfile(batch_size, fileunit)
        read(fileunit, *) self % dropout_rate
    end function dropout_layer_fromfile

end module fnn
