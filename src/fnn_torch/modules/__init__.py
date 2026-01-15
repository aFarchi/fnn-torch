
from fnn_torch.modules.wrapped_module import WrappedModule
from fnn_torch.modules.layer_linear import ExtendedLinear
from fnn_torch.modules.layer_sequential import ExtendedSequential
from fnn_torch.modules.layer_relu import ExtendedReLU
from fnn_torch.modules.layer_tanh import ExtendedTanh
from fnn_torch.modules.layer_skip_connection import SkipConnection


def construct_module(name):
    match name:
        case 'linear':
            master_layer = ExtendedLinear(input_size=8, output_size=4)
            return WrappedModule(master_layer)
        case 'sequential':
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                ExtendedLinear(input_size=4, output_size=2),
            )
            return WrappedModule(master_layer)
        case 'relu':
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                ExtendedReLU(input_size=4),
            )
            return WrappedModule(master_layer)
        case 'tanh':
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                ExtendedTanh(input_size=4),
            )
            return WrappedModule(master_layer)
        case 'skip':
            master_layer = SkipConnection(
                ExtendedLinear(input_size=8, output_size=2),
            )
            return WrappedModule(master_layer)
        case _:
            raise ValueError(f'unknown module: {name}')
