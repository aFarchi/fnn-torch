import torch

class ExtendedGELU(torch.nn.GELU):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.num_parameters = 0

    def save_architecture(self, f):
        f.write('gelu_activation\n')
        f.write(f'{self.input_size}\n')

    @staticmethod
    def get_p_list():
        return ()

    @staticmethod
    def to_named_p(p, prefix):
        return {}

    @staticmethod
    def to_named_dp(dp, prefix):
        return {}

    @staticmethod
    def to_dp_list(named_p, prefix):
        return ()
