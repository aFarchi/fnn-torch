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

    def get_flat_parameters(self):
        return (
            self.bias.detach().numpy(),
            self.weight.detach().numpy().T.flatten(),
        )

    def to_named_parameters(self, p, prefix=''):
        return {
            f'{prefix}bias': p[:self.output_size],
            f'{prefix}weight': p[self.output_size:].reshape(self.input_size, self.output_size).T,
        }

    @staticmethod
    def get_flat_named_parameters(named_p, prefix=''):
        return (
            named_p[f'{prefix}bias'],
            named_p[f'{prefix}weight'].T.flatten(),
        )
