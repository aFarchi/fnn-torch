import torch

class ExtendedLinear(torch.nn.Linear):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.num_parameters = self.output_size * (self.input_size + 1)

    def save_architecture(self, f):
        f.write('linear\n')
        f.write('not-frozen\n')
        f.write(f'{self.input_size}\n')
        f.write(f'{self.output_size}\n')

    def get_p_list(self):
        return (
            self.bias.detach().numpy(),
            self.weight.detach().numpy().T.flatten(),
        )

    def to_named_p(self, p, prefix):
        return {
            f'{prefix}bias': p[:self.output_size],
            f'{prefix}weight': p[self.output_size:].reshape(self.input_size, self.output_size).T,
        }

    def to_named_dp(self, dp, prefix):
        return self.to_named_p(dp, prefix)

    @staticmethod
    def to_dp_list(named_dp, prefix):
        return (
            named_dp[f'{prefix}bias'],
            named_dp[f'{prefix}weight'].T.flatten(),
        )


class FrozenExtendedLinear(torch.nn.Linear):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.num_parameters = 0

    def save_architecture(self, f):
        f.write('linear\n')
        f.write('frozen\n')
        f.write(f'{self.input_size}\n')
        f.write(f'{self.output_size}\n')

    def get_p_list(self):
        return (
            self.bias.detach().numpy(),
            self.weight.detach().numpy().T.flatten(),
        )

    def to_named_p(self, p, prefix):
        return {
            f'{prefix}bias': torch.clone(self.bias),
            f'{prefix}weight': torch.clone(self.weight),
        }

    def to_named_dp(self, dp, prefix):
        return {
            f'{prefix}bias': torch.zeros_like(self.bias),
            f'{prefix}weight': torch.zeros_like(self.weight),
        }

    @staticmethod
    def to_dp_list(named_dp, prefix):
        return ()
