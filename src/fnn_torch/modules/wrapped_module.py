from torch.func import functional_call, jvp, vjp

class WrappedModule:

    def __init__(self, model):
        self.model = model
        self.x = None
        self.p = None
        self.named_p = None

    def named_forward(self, named_p, x):
        return functional_call(self.model, named_p, x)

    def forward(self, p, x):
        x = x.reshape(self.model.input_shape)
        self.p = p
        self.x = x
        self.named_p = self.model.to_named_parameters(p)
        y = self.named_forward(self.named_p, self.x)
        return y.reshape(-1)

    def apply_tl(self, dp, dx):
        dx = dx.reshape(self.model.input_shape)
        named_dp = self.model.to_named_parameters(dp)
        return jvp(self.named_forward, (self.named_p, self.x), (named_dp, dx))[1].reshape(-1)

    def apply_ad(self, dy):
        dy = dy.reshape(self.model.output_shape)
        _, vjp_fn = vjp(self.named_forward, self.named_p, self.x)
        dp, dx = vjp_fn(dy)
        dp_flat = self.model.flatten_named_parameters(dp)
        return dp_flat, dx.reshape(-1)
