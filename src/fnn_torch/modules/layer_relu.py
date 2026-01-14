import torch

class ExtendedReLU(torch.nn.ReLU):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.num_parameters = 0

    def save_architecture(self, f):
        f.write('relu_activation\n')
        f.write(f'{self.input_size}\n')

    @staticmethod
    def get_flat_parameters():
        return ()

    @staticmethod
    def to_named_parameters(p, prefix=''):
        return {}

    @staticmethod
    def get_flat_named_parameters(named_p, prefix=''):
        return ()
