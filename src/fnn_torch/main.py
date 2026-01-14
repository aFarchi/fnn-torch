import logging

import rich.logging
import rich_click as click

logger = logging.getLogger(__name__)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    logging.basicConfig(
        level='INFO',
        format='%(message)s',
        datefmt='[%X]',
        handlers=[rich.logging.RichHandler()],
    )


@cli.command(name='init')
@click.argument(
    'name',
    default='linear',
    help='name of the model to initialise (default "linear")',
)
@click.argument(
    'encoding',
    default='f8',
    help='encoding used for binary files (default "f8")',
)
def init(name, encoding):
    """Initialises the NN before the fortran run."""
    from fnn_torch.init_export import init_and_export
    init_and_export(name, encoding)


@cli.command(name='check')
@click.argument(
    'encoding',
    default='f8',
    help='encoding used for binary files (default "f8")',
)
def check(encoding):
    """Checks the output of the fortran run."""
    from fnn_torch.check import check
    check(encoding)


if __name__ == '__main__':
    cli()
