import logging
import random

import numpy as np
import torch

from fnn_torch.wdir import wdir
from fnn_torch.modules import construct_module

logger = logging.getLogger(__name__)


def init_and_export(name, encoding):
    random.seed(3141592)
    np.random.seed(3141592)
    torch.manual_seed(3141592)
    torch.set_default_dtype(torch.float64)

    logger.info(f'creating "{wdir}"')
    wdir.mkdir(exist_ok=True)

    logger.info(f'creating model "{name}"')
    model = construct_module(name)

    filename = wdir / 'architecture.txt'
    logger.info(f'saving architecture into "{filename}"')
    model.save_architecture(filename)

    filename = wdir / 'parameters.bin'
    logger.info(f'saving parameters into "{filename}"')
    logger.info(f'using encoding "{encoding}"')
    model.save_parameters(filename, encoding=encoding)

    filename = wdir / 'config.toml'
    logger.info(f'saving config into "{filename}"')
    with open(filename, 'w') as f:
        f.write(f'name = "{name}"\n')
