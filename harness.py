#!/usr/bin/env python

from zensols import deepnlp


# initialize the NLP system
deepnlp.init()


if (__name__ == '__main__'):
    from zensols.cli import ConfigurationImporterCliHarness
    lang: str = {
        0: 'en',
        1: 'sq',
    }[1]
    config: str = {
        0: 'wordvec',
        1: 'transformer',
    }[0]
    action: str = {
        0: 'proto',
        1: 'debug',
        2: 'traintest',
        3: 'info -i model',
        4: 'finecompile',
        5: 'finestats',
        6: 'finedump',
    }[0]
    ConfigurationImporterCliHarness(
        src_dir_name='src',
        package_resource='uic.edusenti',
        config_path=f'models/{config}.yml',
        proto_args=f'{action} --override edusenti_default.lang={lang}',
        proto_factory_kwargs={'reload_pattern': r'^uic.edusenti'},
    ).run()
