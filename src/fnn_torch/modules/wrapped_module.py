import numpy as np
import torch

from torch.func import functional_call, jvp, vjp

class WrappedModule(torch.nn.Module):

    def __init__(self, master_layer):
        super().__init__()
        self.master_layer = master_layer
        self.batch_size = 16
        self.input_shape = (self.batch_size, self.master_layer.input_size)
        self.output_shape = (self.batch_size, self.master_layer.output_size)
        self.x = None
        self.p = None
        self.named_p = None

    def regular_forward(self, x):
        self.x = None
        self.p = None
        self.named_p = None
        x = x.reshape(self.input_shape)
        y = self.master_layer(x)
        return y.reshape(-1)

    def named_forward(self, named_p, x):
        return functional_call(self.master_layer, named_p, x)

    def forward(self, p, x):
        x = x.reshape(self.input_shape)
        self.p = p
        self.x = x
        self.named_p = self.to_named_p(p)
        y = self.named_forward(self.named_p, self.x)
        return y.reshape(-1)

    def apply_tl(self, dp, dx):
        dx = dx.reshape(self.input_shape)
        named_dp = self.to_named_dp(dp)
        return jvp(self.named_forward, (self.named_p, self.x), (named_dp, dx))[1].reshape(-1)

    def apply_ad(self, dy):
        dy = dy.reshape(self.output_shape)
        _, vjp_fn = vjp(self.named_forward, self.named_p, self.x)
        named_dp, dx = vjp_fn(dy)
        dp = self.to_dp(named_dp)
        return dp, dx.reshape(-1)

    def save_architecture(self, filename):
        with open(filename, 'w') as f:
            self.master_layer.save_architecture(f)

    def save_parameters(self, filename, encoding=None):
        parameters = self.master_layer.get_p_list()
        parameters = np.concat(parameters)
        if encoding is not None:
            parameters = parameters.astype(encoding)
        parameters.tofile(filename)

    def to_named_p(self, p):
        return self.master_layer.to_named_p(p, prefix='')

    def to_named_dp(self, dp):
        return self.master_layer.to_named_dp(dp, prefix='')

    def to_dp(self, named_dp):
        dp_list = self.master_layer.to_dp_list(named_dp, prefix='')
        return torch.cat(dp_list)
