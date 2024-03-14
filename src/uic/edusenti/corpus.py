"""Contains container and utility classes to parse read the corpus.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
from pathlib import Path
import pandas as pd
from zensols.dataframe import ResourceFeatureDataframeStash
from zensols.deepnlp.classify import LabeledFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class SentimentFeatureDocument(LabeledFeatureDocument):
    """A feature document that contains the topic (i.e. subject) and emotion
    (i.e. joy, fear, etc) of the corresponding sentence(s).  This document
    usually has one sentence per the corpus, but can have more if the language
    parser chunks it as such.

    """
    topic: str = field(default=None)
    """The subject of the review (i.e. project, instruction, general, etc)."""

    emotion: str = field(default=None)
    """The emotion of the reveiw (i.e. joy, fear, surpise, etc)."""

    def __post_init__(self):
        super().__post_init__()
        # the corpus contains a sentence for each row/data point, so copy this
        # to all sentences (see class docs)
        for sent in self.sents:
            sent.topic = self.topic
            sent.emotion = self.emotion

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write()
        self._write_line(f'topic: {self.topic}', depth + 1, writer)
        self._write_line(f'emotion: {self.emotion}', depth + 1, writer)


@dataclass
class SentimentDataframeStash(ResourceFeatureDataframeStash):
    """Create the dataframe by reading the sentiment sentences from the corpus
    files.

    """
    lang: str = field()
    """The corpus language."""

    labels: Tuple[str] = field()
    """The labels of the classification, which are::

      * ``+``: positive sentiment
      * ``-``: negative sentiment
      * ``n``: neutral sentiment

    """
    def _get_dataframe(self) -> pd.DataFrame:
        self.installer()
        corp_dir: Path = self.installer.get_singleton_path()
        corp_file: Path = corp_dir / f'{self.lang}.csv'
        return pd.read_csv(corp_file)
