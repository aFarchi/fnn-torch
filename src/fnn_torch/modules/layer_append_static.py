import torch

class AppendStaticInput(torch.nn.Module):

    def __init__(self, input_size, x_extra):
        super().__init__()
        self.register_parameter(
            name='x_extra',
            param=torch.nn.Parameter(torch.from_numpy(x_extra)),
        )
        self.input_size = input_size
        self.output_size = input_size + self.x_extra.numel()
        self.num_parameters = self.x_extra.numel()

    def forward(self, x):
        x_extra = self.x_extra
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            x_extra = x_extra.reshape((1, self.num_parameters))
            x_extra = x_extra.expand((batch_size, self.num_parameters))
        return torch.cat((x, x_extra), dim=-1)

    def save_architecture(self, f):
        f.write('append_static_input\n')
        f.write('not-frozen\n')
        f.write(f'{self.input_size}\n')
        f.write(f'{self.num_parameters}\n')

    def get_p_list(self):
        return (
            self.x_extra.detach().numpy(),
        )

    @staticmethod
    def to_named_p(p, prefix):
        return {
            f'{prefix}x_extra': p,
        }

    @staticmethod
    def to_named_dp(dp, prefix):
        return {
            f'{prefix}x_extra': dp,
        }

    @staticmethod
    def to_dp_list(named_dp, prefix):
        return (
            named_dp[f'{prefix}x_extra'],
        )
