import torch

class ExtendedSequential(torch.nn.Sequential):

    @property
    def input_size(self):
        return self[0].input_size

    @property
    def output_size(self):
        return self[-1].output_size

    @property
    def num_parameters(self):
        return sum(layer.num_parameters for layer in self)

    def save_architecture(self, f):
        f.write('sequential\n')
        f.write(f'{len(self)}\n')
        for layer in self:
            layer.save_architecture(f)

    def get_flat_parameters(self):
        flat_parameters = []
        for layer in self:
            flat_parameters.extend(layer.get_flat_parameters())
        return flat_parameters

    def to_named_parameters(self, p, prefix=''):
        named_parameters = {}
        index = 0
        for (i, layer) in enumerate(self):
            named_parameters.update(layer.to_named_parameters(
                p[index:index + layer.num_parameters],
                prefix=f'{prefix}{i}.',
            ))
            index += layer.num_parameters
        return named_parameters

    def get_flat_named_parameters(self, named_p, prefix=''):
        flat_parameters = []
        for (i, layer) in enumerate(self):
            flat_parameters.extend(layer.get_flat_named_parameters(
                named_p,
                prefix=f'{prefix}{i}.',
            ))
        return flat_parameters
