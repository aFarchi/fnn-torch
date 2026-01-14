import logging
import random
import tomllib

import numpy as np
import torch

from fnn_torch.wdir import wdir
from fnn_torch.modules import construct_module
from fnn_torch.modules.wrapped_module import WrappedModule
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
    wrapped_model = WrappedModule(model)

    logger.info('loading data')
    logger.info(f'using encoding "{encoding}"')
    x = load_tensor(wdir / 'x.bin', encoding)
    y_fnn = load_tensor(wdir / 'y.bin', encoding)

    logger.info('applying forward')
    y_py = model(x.reshape(model.input_shape)).reshape(-1)
    compare_tensors('forward', y_fnn, y_py, rtol=1e-8, atol=0)
