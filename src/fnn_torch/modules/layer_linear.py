import numpy as np
import torch


class LinearLayer(torch.nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.input_shape = (batch_size, 8)
        self.output_shape = (batch_size, 4)

    def forward(self, x):
        x = self.linear(x)
        return x

    def save_architecture(self, filename):
        with open(filename, 'w') as f:
            f.write('linear\n')
            f.write('not-frozen\n')
            f.write(f'{self.input_shape[1]}\n')
            f.write(f'{self.output_shape[1]}\n')

    def save_parameters(self, filename, encoding=None):
        parameters = np.concat((
            self.linear.bias.detach().numpy(),
            self.linear.weight.detach().numpy().T.flatten(),
        ))
        if encoding is not None:
            parameters = parameters.astype(encoding)
        parameters.tofile(filename)

    def to_named_parameters(self, p):
        input_size = self.input_shape[1]
        output_size = self.output_shape[1]
        named_parameters = {
            'linear.bias': p[:output_size],
            'linear.weight': p[output_size:].reshape(input_size, output_size).T,
        }
        return named_parameters

    @staticmethod
    def flatten_named_parameters(named_p):
        return torch.cat((
            named_p['linear.bias'],
            named_p['linear.weight'].T.flatten(),
        ))
