import torch
import numpy as np

class RegionSplit(torch.nn.Module):

    def __init__(self, layer_sh, layer_tr, layer_nh, index_sin_lat):
        super().__init__()
        self.layer_sh = layer_sh
        self.layer_tr = layer_tr
        self.layer_nh = layer_nh
        self.index_sin_lat = index_sin_lat
        self.sin_30 = float(np.sin(np.radians(30)))
        self.sin_20 = float(np.sin(np.radians(20)))
        self.input_size = layer_sh.input_size
        self.output_size = layer_sh.output_size
        self.num_parameters = layer_sh.num_parameters + layer_tr.num_parameters + layer_nh.num_parameters

    def forward(self, x):
        sin_lat = x[..., self.index_sin_lat:self.index_sin_lat + 1]
        alpha_nh = (sin_lat - self.sin_20) / (self.sin_30 - self.sin_20)
        alpha_nh = torch.clamp(alpha_nh, 0, 1)
        alpha_sh = -(sin_lat + self.sin_20) / (self.sin_30 - self.sin_20)
        alpha_sh = torch.clamp(alpha_sh, 0, 1)
        alpha_tr = (1 - alpha_nh) * (1 - alpha_sh)
        y_sh = self.layer_sh(x)
        y_tr = self.layer_tr(x)
        y_nh = self.layer_nh(x)
        return alpha_nh * y_nh + alpha_tr * y_tr + alpha_sh * y_sh

    def save_architecture(self, f):
        f.write('region_split\n')
        f.write(f'{self.index_sin_lat + 1}\n')
        self.layer_sh.save_architecture(f)
        self.layer_tr.save_architecture(f)
        self.layer_nh.save_architecture(f)

    def get_p_list(self):
        return self.layer_sh.get_p_list() + self.layer_tr.get_p_list() + self.layer_nh.get_p_list()

    def to_named_p(self, p, prefix):
        named_p = {}
        index = 0
        for label, layer in [
            ('sh', self.layer_sh),
            ('tr', self.layer_tr),
            ('nh', self.layer_nh),
        ]:
            named_p.update(layer.to_named_p(
                p[index:index + layer.num_parameters],
                prefix=f'{prefix}layer_{label}.',
            ))
            index += layer.num_parameters
        return named_p

    def to_named_dp(self, dp, prefix):
        named_dp = {}
        index = 0
        for label, layer in [
            ('sh', self.layer_sh),
            ('tr', self.layer_tr),
            ('nh', self.layer_nh),
        ]:
            named_dp.update(layer.to_named_dp(
                dp[index:index + layer.num_parameters],
                prefix=f'{prefix}layer_{label}.',
            ))
            index += layer.num_parameters
        return named_dp

    def to_dp_list(self, named_dp, prefix):
        dp_list = []
        for label, layer in [
            ('sh', self.layer_sh),
            ('tr', self.layer_tr),
            ('nh', self.layer_nh),
        ]:
            dp_list.extend(layer.to_dp_list(
                named_dp,
                prefix=f'{prefix}layer_{label}.',
            ))
        return dp_list
    