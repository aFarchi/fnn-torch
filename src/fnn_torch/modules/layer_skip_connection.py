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

    def get_p_list(self):
        return self.internal_layer.get_p_list()

    def to_named_p(self, p, prefix):
        return self.internal_layer.to_named_p(
            p,
            prefix=f'{prefix}internal_layer.',
        )

    def to_named_dp(self, dp, prefix):
        return self.internal_layer.to_named_dp(
            dp,
            prefix=f'{prefix}internal_layer.',
        )

    def to_dp_list(self, named_dp, prefix):
        return self.internal_layer.to_dp_list(
            named_dp,
            prefix=f'{prefix}internal_layer.',
        )
