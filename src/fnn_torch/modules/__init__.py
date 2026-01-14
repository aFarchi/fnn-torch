
def construct_module(name):
    batch_size = 16
    match name:
        case 'linear':
            from fnn_torch.modules.layer_linear import LinearLayer
            return LinearLayer(batch_size)
        case _:
            raise ValueError(f'unknown module: {name}')
