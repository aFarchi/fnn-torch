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

    def get_p_list(self):
        flat_parameters = []
        for layer in self:
            flat_parameters.extend(layer.get_p_list())
        return flat_parameters

    def to_named_p(self, p, prefix):
        named_p = {}
        index = 0
        for (i, layer) in enumerate(self):
            named_p.update(layer.to_named_p(
                p[index:index + layer.num_parameters],
                prefix=f'{prefix}{i}.',
            ))
            index += layer.num_parameters
        return named_p

    def to_named_dp(self, dp, prefix):
        named_dp = {}
        index = 0
        for (i, layer) in enumerate(self):
            named_dp.update(layer.to_named_dp(
                dp[index:index + layer.num_parameters],
                prefix=f'{prefix}{i}.',
            ))
            index += layer.num_parameters
        return named_dp

    def to_dp_list(self, named_dp, prefix):
        dp_list = []
        for (i, layer) in enumerate(self):
            dp_list.extend(layer.to_dp_list(
                named_dp,
                prefix=f'{prefix}{i}.',
            ))
        return dp_list
