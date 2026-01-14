
from fnn_torch.modules.wrapped_module import WrappedModule

def construct_module(name):
    match name:
        case 'linear':
            from fnn_torch.modules.layer_linear import ExtendedLinear
            master_layer = ExtendedLinear(input_size=8, output_size=4)
            return WrappedModule(master_layer)
        case 'sequential':
            from fnn_torch.modules.layer_sequential import ExtendedSequential
            from fnn_torch.modules.layer_linear import ExtendedLinear
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                ExtendedLinear(input_size=4, output_size=2),
            )
            return WrappedModule(master_layer)
        case _:
            raise ValueError(f'unknown module: {name}')
