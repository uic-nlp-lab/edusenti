"""Environemnt configuration and set up: add this (deepnlp) library to the
Python path and framework entry point.

"""
__author__ = 'Paul Landes'

from typing import List
from pathlib import Path


class JupyterManagerFactory(object):
    """Bootstrap and import libraries to automate notebook testing.

    """
    def __init__(self, app_root_dir: Path = Path('..')):
        """Set up the interpreter environment so we can import local packages.

        :param app_root_dir: the application root directory

        """
        from zensols import deepnlp
        deepnlp.init()
        from zensols.cli import ConfigurationImporterCliHarness
        self._harness = ConfigurationImporterCliHarness(
            src_dir_name='src',
            package_resource='uic.edusenti',
            root_dir=app_root_dir)

    def __call__(self):
        """Create a new ``JupyterManager`` instance and return it."""
        from zensols.deeplearn.cli import JupyterManager

        def map_args(model: str = None, **kwargs):
            """Convert args to override string.

            :param kwargs: arguments include: ``lang``, ``name``

            """
            args: List[str] = []
            sec: str = 'edusenti_default'
            if len(kwargs) > 0:
                ostr = ','.join(map(lambda kv: f'{sec}.{kv[0]}={kv[1]}',
                                    kwargs.items()))
                args.extend(['--override', ostr])
            if model is not None:
                args.extend(['--config', f'../models/{model}.yml'])
            return args

        return JupyterManager(self._harness, cli_args_fn=map_args,
                              reduce_logging=True)
