import torch

class SkipConnection(torch.nn.Module):

    def __init__(self, internal_layer):
        super().__init__()
        self.internal_layer = internal_layer
        self.input_size = internal_layer.input_size
        self.output_size = internal_layer.input_size + internal_layer.output_size
        self.num_parameters = internal_layer.num_parameters

    def forward(self, x):
        y = self.internal_layer(x)
        return torch.cat((x, y), dim=-1)

    def save_architecture(self, f):
        f.write('skip_connection\n')
        self.internal_layer.save_architecture(f)

    def get_flat_parameters(self):
        return self.internal_layer.get_flat_parameters()

    def to_named_parameters(self, p, prefix=''):
        return self.internal_layer.to_named_parameters(
            p,
            prefix=f'{prefix}internal_layer.',
        )

    def get_flat_named_parameters(self, named_p, prefix=''):
        return self.internal_layer.get_flat_named_parameters(
            named_p,
            prefix=f'{prefix}internal_layer.',
        )
