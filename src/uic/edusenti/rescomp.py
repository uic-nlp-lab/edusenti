"""Compile the results for the paper.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Type
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score as cohen_kappa
from zensols.persist import persisted
from zensols.db import DbPersister
from zensols.datdesc import DataFrameDescriber, DataDescriber
from . import Corpus


@dataclass
class ResultCompiler(object):
    corpus: Corpus = field()
    double_anon_file: Path = field()

    @property
    @persisted('_corpus_stats', cache_global=True)
    def corpus_stats(self) -> DataFrameDescriber:
        per: DbPersister = self.corpus.persister
        qs = (('sentences', 'select_sent_count'),
              ('tokens', 'select_token_count'),
              ('characters', 'select_char_count'))
        rows: List[Tuple[str, str]] = []
        name: str
        sql: str
        for name, sql in qs:
            res: int = per.execute_singleton_by_name(sql)[0]
            rows.append((name, res))
        return DataFrameDescriber(
            name='Corpus Statistics',
            df=pd.DataFrame(rows, columns='name count'.split()),
            desc='Fine-tuned Albanian Statistics Trained on Masked Tokens',
            meta=(('name', 'Description'),
                  ('count', 'Count')))

    @property
    @persisted('_corpus_sources', cache_global=True)
    def corpus_sources(self) -> DataFrameDescriber:
        return DataFrameDescriber(
            name='Corpus Sources',
            df=self.corpus.persister.execute_by_name(
                'select_corpus_sources', row_factory='pandas'),
            desc='Sources of the Fine-tuned Albanian Corpus',
            meta=(('name', 'Name'),
                  ('url', 'URL'),
                  ('count', 'Count')))

    @property
    @persisted('_kappa', cache_global=True)
    def kappa(self) -> DataFrameDescriber:
        df: pd.DataFrame = pd.read_csv(self.double_anon_file)
        df = df.drop(columns=['notes']).dropna()
        rows: List[Tuple[str, float]] = [('sentences', len(df))]
        t: Type
        col_name: str
        for t, col_name in ((str, 'topic'), (str, 'emotion'), (int, 'label')):
            ca: str = f'a1_{col_name}'
            cb: str = f'a2_{col_name}'
            ks: float = cohen_kappa(df[ca].astype(t), df[cb].astype(t))
            rows.append((col_name, ks))
        return DataFrameDescriber(
            name='Kappa Scores',
            df=pd.DataFrame(rows, columns='description value'.split()),
            desc="Cohen's Kappa Scores")

    def __call__(self):
        dd = DataDescriber(
            describers=(self.corpus_stats, self.corpus_sources, self.kappa),
            name='EduSenti')
        dd.write()
        dd.save()
