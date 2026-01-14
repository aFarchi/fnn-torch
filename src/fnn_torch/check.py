import logging
import random
import tomllib

import numpy as np
import torch

from fnn_torch.wdir import wdir
from fnn_torch.modules import construct_module
from fnn_torch.io import load_tensor
from fnn_torch.compare import compare_tensors, compare_arrays

logger = logging.getLogger(__name__)


def check(encoding):
    random.seed(3141592)
    np.random.seed(3141592)
    torch.manual_seed(3141592)
    torch.set_default_dtype(torch.float64)

    filename = wdir / 'config.toml'
    logger.info(f'reading config from "{filename}"')
    with open(filename, 'rb') as f:
        config = tomllib.load(f)

    logger.info('creating model')
    model = construct_module(**config)

    logger.info('loading data')
    logger.info(f'using encoding "{encoding}"')
    p = load_tensor(wdir / 'p.bin', encoding)
    x = load_tensor(wdir / 'x.bin', encoding)
    dp = load_tensor(wdir / 'dp.bin', encoding)
    dx = load_tensor(wdir / 'dx.bin', encoding)
    dy = load_tensor(wdir / 'dy.bin', encoding)
    y_fnn = load_tensor(wdir / 'y.bin', encoding)
    py_fnn = load_tensor(wdir / 'py.bin', encoding)
    FpT_dy_fnn = load_tensor(wdir / 'FpT_dy.bin', encoding)
    FxT_dy_fnn = load_tensor(wdir / 'FxT_dy.bin', encoding)
    F_dx_dp_fnn = load_tensor(wdir / 'F_dx_dp.bin', encoding)

    logger.info('applying forward')
    y_py = model.regular_forward(x)
    compare_tensors('forward', y_fnn, y_py, rtol=1e-8, atol=0)

    logger.info('applying parametrised forward')
    y_py = model.forward(p, x)
    compare_tensors('parametrised forward', py_fnn, y_py, rtol=1e-8, atol=0)

    logger.info('applying adjoint')
    ad_py = model.apply_ad(dy)
    FpT_dy_py = ad_py[0]
    FxT_dy_py = ad_py[1]
    compare_tensors('FpT_dy', FpT_dy_fnn, FpT_dy_py, rtol=1e-8, atol=0)
    compare_tensors('FxT_dy', FxT_dy_fnn, FxT_dy_py, rtol=1e-8, atol=0)

    logger.info('applying tangent linear')
    F_dx_dp_py = model.apply_tl(dp, dx)
    compare_tensors('F_dx_dp', F_dx_dp_fnn, F_dx_dp_py, rtol=1e-8, atol=0)

    logger.info('computing adjoint test')
    dot_1 = np.array([FpT_dy_fnn @ dp + FxT_dy_fnn @ dx])
    dot_2 = np.array([F_dx_dp_py @ dy])
    compare_arrays('dot', dot_1, dot_2, rtol=1e-8, atol=0, n_first=1)
