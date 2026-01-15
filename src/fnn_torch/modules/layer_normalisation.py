import torch

class Normalisation(torch.nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.register_parameter(
            name='alpha',
            param=torch.nn.Parameter(torch.from_numpy(alpha)),
        )
        self.register_parameter(
            name='beta',
            param=torch.nn.Parameter(torch.from_numpy(beta)),
        )
        self.input_size = self.alpha.numel()
        self.output_size = self.input_size
        self.num_parameters = 2 * self.input_size

    def forward(self, x):
        return self.alpha * x + self.beta

    def save_architecture(self, f):
        f.write('normalisation\n')
        f.write('not-frozen\n')
        f.write(f'{self.input_size}\n')

    def get_p_list(self):
        return (
            self.alpha.detach().numpy(),
            self.beta.detach().numpy(),
        )

    def to_named_p(self, p, prefix):
        return {
            f'{prefix}alpha': p[:self.input_size],
            f'{prefix}beta': p[self.input_size:],
        }

    def to_named_dp(self, dp, prefix):
        return {
            f'{prefix}alpha': dp[:self.input_size],
            f'{prefix}beta': dp[self.input_size:],
        }

    @staticmethod
    def to_dp_list(named_dp, prefix):
        return (
            named_dp[f'{prefix}alpha'],
            named_dp[f'{prefix}beta'],
        )
