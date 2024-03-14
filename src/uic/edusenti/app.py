"""Student to instructor review sentiment analysis prototyping.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from pathlib import Path
from zensols.config import ConfigFactory
from zensols.cli import Cleaner
from . import (
    CorpusCompiler, CorpusDumper, Corpus,
    TokenizerTrainer, MaskedModelTrainer,
)

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Student to instructor review sentiment analysis.

    """
    config_factory: ConfigFactory = field()
    """Creates this instance and provides prototyping."""

    def _rm(self):
        cleaner = self.config_factory('cleaner_cli')
        cleaner.clean_level = 2
        cleaner()

    def _test_doc_parser(self):
        stash = self.config_factory('dataframe_stash')
        sent = next(iter(stash.values()))['text']
        dp = self.config_factory('doc_parser')
        doc = dp(sent)
        for t in doc.tokens:
            t.write()

    def _test_feat_stash(self):
        import itertools as it
        stash = self.config_factory('feature_factory_stash')
        print('len', len(stash))
        for k, v in it.islice(stash, 1):
            print(k, v)
            print('T', v.topic)

        for v in stash.values():
            if v.topic is None:
                print(v, v.topic)

    def _test_batch_stash(self):
        import itertools as it
        stash = self.config_factory('batch_stash')
        print('len', len(stash))
        for k, v in it.islice(stash, 1):
            print(k, type(v))
            v.write()


@dataclass
class FineTuneApplication(object):
    """Albanian fine tuning utilities.

    """
    config_factory: ConfigFactory = field()

    corpus_compiler: CorpusCompiler = field()
    """The fine tune Albanian corpus compiler in to an SQLite file."""

    corpus_dumper: CorpusDumper = field()
    """Dumps the cleaned corpus in memory or on the disk as CSV files. """

    cleaner: Cleaner = field()

    corpus: Corpus = field()
    """The manager that reads (and writes) the dataset from the file system."""

    tokenizer_trainer: TokenizerTrainer = field()
    """Trains the model."""

    model_trainer: MaskedModelTrainer = field()
    """Trains the model."""

    def compile_finetuned(self):
        """Compile the fine tuned corpus in to an SQLite file.

        """
        print('adding to sqlite')
        self.corpus_compiler.compile(force=True)

    def print_stats(self):
        """Print statistics."""
        print(self.corpus_compiler.stats_df)

    def dump(self, out_dir: Path = Path('sq-finetune-corpus')):
        """Write the finetuned cleaned corpus to the file system.

        :param out_dir: where to write the CSV files

        """
        self.corpus_dumper.dump(out_dir)

    def compile_dataset(self):
        """Transfer the data from the SQLite corpus file to a dataset on disk
        (needed for training).

        """
        self.corpus.write_dataset()

    def clear_models(self):
        """Delete previously finetuned trained models.

        """
        self.cleaner()

    def train_tokenizer(self):
        """Train the tokenizer.

        """
        self.tokenizer_trainer.train()

    def train_model(self):
        """Train the model.

        """
        self.model_trainer.train()

    def write_corpus_stats(self):
        """Write the corpus statistics to the file system.

        """
        comp = self.config_factory('edusenti_result_compiler')
        comp()

    def proto(self, run: int = 0):
        """Used for prototyping.

        """
        {1: self.print_stats,
         2: self.compile_dataset,
         3: self.train_tokenizer,
         4: self.train_model,
         }[run]()
