import numpy as np

from fnn_torch.modules.layer_gelu import ExtendedGELU
from fnn_torch.modules.wrapped_module import WrappedModule
from fnn_torch.modules.layer_linear import ExtendedLinear, FrozenExtendedLinear
from fnn_torch.modules.layer_sequential import ExtendedSequential
from fnn_torch.modules.layer_relu import ExtendedReLU
from fnn_torch.modules.layer_tanh import ExtendedTanh
from fnn_torch.modules.layer_skip_connection import SkipConnection
from fnn_torch.modules.layer_normalisation import Normalisation, FrozenNormalisation
from fnn_torch.modules.layer_append_static import AppendStaticInput, FrozenAppendStaticInput
from fnn_torch.modules.layer_dropout import ExtendedDropout


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
        case 'frozen-linear':
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                FrozenExtendedLinear(input_size=4, output_size=2),
            )
            return WrappedModule(master_layer)
        case 'relu':
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                ExtendedReLU(input_size=4),
            )
            return WrappedModule(master_layer)
        case 'gelu':
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                ExtendedGELU(input_size=4),
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
        case 'normalisation':
            alpha = 1 + 0.1 * np.random.randn(4)
            beta = np.random.randn(4)
            master_layer = Normalisation(alpha, beta)
            return WrappedModule(master_layer)
        case 'frozen-normalisation':
            alpha = 1 + 0.1 * np.random.randn(4)
            beta = np.random.randn(4)
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                FrozenNormalisation(alpha, beta),
            )
            return WrappedModule(master_layer)
        case 'append':
            x_extra = np.random.randn(4)
            master_layer = AppendStaticInput(input_size=8, x_extra=x_extra)
            return WrappedModule(master_layer)
        case 'frozen-append':
            x_extra = np.random.randn(4)
            master_layer = ExtendedSequential(
                ExtendedLinear(input_size=8, output_size=4),
                FrozenAppendStaticInput(input_size=4, x_extra=x_extra),
            )
            return WrappedModule(master_layer)
        case 'dropout':
            alpha = np.zeros(10)
            beta = np.ones(10)
            master_layer = ExtendedSequential(
                Normalisation(alpha, beta),
                ExtendedDropout(input_size=10, dropout_rate=0.25),
            )
            return WrappedModule(master_layer)
        case 'r03':
            alpha_in = np.ones(420)
            beta_in = np.zeros(420)
            alpha_out = np.ones(412)
            beta_out = np.zeros(412)
            master_layer = ExtendedSequential(
                FrozenNormalisation(alpha_in, beta_in),
                ExtendedLinear(input_size=420, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=412),
                FrozenNormalisation(alpha_out, beta_out),
            )
            return WrappedModule(master_layer)
        case 'r12q':
            alpha_in = np.ones(557)
            beta_in = np.zeros(557)
            alpha_out = np.ones(549)
            beta_out = np.zeros(549)
            master_layer = ExtendedSequential(
                FrozenNormalisation(alpha_in, beta_in),
                ExtendedLinear(input_size=557, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=512),
                ExtendedTanh(input_size=512),
                ExtendedLinear(input_size=512, output_size=549),
                FrozenNormalisation(alpha_out, beta_out),
            )
            return WrappedModule(master_layer)
        case _:
            raise ValueError(f'unknown module: {name}')
