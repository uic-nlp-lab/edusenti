"""Compile the Albanian pretraining corpus into an SQLite database for fine
tuning tasks.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Iterable, Tuple, List, Any, Optional, Set, ClassVar, Dict
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import sys
import os
import logging
import random
import re
import json
import itertools as it
from io import TextIOBase, StringIO
from pathlib import Path
from lxml import etree
from lxml.etree import _Element as Element
import pandas as pd
import langdetect as ld
import wikitextparser as wtp
import pyarrow as pa
from pyarrow import Table
from datasets import Dataset
from zensols.config import Dictable
from zensols.persist import persisted, Stash, ReadOnlyStash, chunks
from zensols.db import DbPersister, cursor
from zensols.nlp import FeatureToken, FeatureSentence, FeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class XmlResource(object):
    """Container for the XML file and the parsed data's root.

    """
    _NS: ClassVar[str] = {'w': 'http://www.mediawiki.org/xml/export-0.10/'}

    path: Path = field()
    """The XML file."""

    @property
    @persisted('_root')
    def root(self):
        """The parsed data's root."""
        with open(self.path, 'rb') as f:
            return etree.parse(f)

    def get_elems(self, file_name: str, xpath: str) -> List[Element]:
        """Return a list of elements matching an XPath.

        :param file_name: the file name in the MS Word ``docx`` zip formatted
                          file

        :param xpath: the xpath matching elements to return

        """
        et: Element = self.root
        els: List[Element] = et.xpath(f'//w:{xpath}', namespaces=self._NS)
        return els

    def xpath(self, *args, **kwargs):
        """Return a list of nodes matching the xpath given as the arguments
        passed to :meth:`lxml.etree.xpath`.

        """
        return self.root.xpath(*args, **kwargs, namespaces=self._NS)


@dataclass(eq=True)
class Sentence(Dictable):
    """A corpus sentence."""

    tokens: Tuple[str] = field()
    """The tokens parsed from the XML file."""

    sid: Optional[str] = field(default=None)
    """The sentence ID if there is one."""

    @property
    def text(self) -> str:
        """The text of the sentence, which is a string of space delimited
        tokens.

        """
        return ' '.join(self.tokens)

    @property
    def span_str(self) -> str:
        """The Python tuple formatted token extents."""
        toks: Tuple[str] = self.tokens
        val: str
        if len(toks) <= 1:
            val = None
        else:
            sio = StringIO()
            st: int = 0
            tok: str
            for tok in toks:
                en: int = st + len(tok)
                sio.write(f'({st},{en}),')
                st = en + 1
            val = sio.getvalue()[:-1]
        return val

    @classmethod
    def from_span_str(cls, text: str, span_str: str,
                      sid: str, corpus_id: int = None) -> Sentence:
        """Create an instance from DB row data.

        :param text: the text of the sentence

        :param span_str: the Python tuple formatted token extents

        :param sid: the original corpus ID

        :param corpus_id: the SQLite DB sentence table ID

        """
        if span_str is None:
            toks = (text,)
        else:
            toks = tuple(map(lambda s: text[s[0]:s[1]], eval(span_str)))
        inst = cls(toks, sid)
        if corpus_id is not None:
            inst.corpus_id = corpus_id
        return inst

    def to_feature_sentence(self) -> FeatureSentence:
        """Create a new feature sentence from this instance.

        :return: the new instance with an ``corpus_id`` attribute

        """
        ftoks: List[FeatureToken] = []
        st: int = 0
        tok: str
        for i, tok in enumerate(self.tokens):
            en: int = st + len(tok)
            ftok = FeatureToken(i=i, idx=st, i_sent=i, norm=tok)
            ftoks.append(ftok)
            st = en + 1
        sent = FeatureSentence(tokens=tuple(ftoks), text=self.text)
        sent.corpus_id = self.corpus_id
        return sent

    def __str__(self) -> str:
        return f'{self.sid}: {self.text}'


@dataclass
class Document(Dictable):
    """A corpus document contains all the sentences from one XML corpus file.

    """
    path: Path = field()
    """The file whence the data originated."""

    sents: Tuple[Sentence] = field()
    """The sentences parsed from the XML file."""

    def to_feature_document(self) -> FeatureDocument:
        """Create a new feature document from this instance.

        """
        doc = FeatureDocument(
            sents=tuple(map(Sentence.to_feature_sentence, self.sents)))
        return doc

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'path: {self.path}', depth, writer)
        for sent in self.sents:
            self._write_line(f'{sent}', depth + 1, writer)

    def __len__(self) -> int:
        return len(self.sents)


@dataclass
class DocumentFactory(object, metaclass=ABCMeta):
    """Parses the corpus files and returns them as :class:`.Document` data
    instances.

    """
    _SENT_REGEX: ClassVar[re.Pattern] = re.compile(r'([^\.!?]*[\.!?])', re.M)
    _TOK_REGEX: ClassVar[re.Pattern] = re.compile(
        r'(?:[\\(){}[\]=&|^+<>/*%;.\'"?!~-]|(?:\w+|\d+))')

    file_glob: str = field()
    """The file name pattern used to find XML files to parse corpus data."""

    corpus_path_offset: int = field()
    """The number of path components before where the corpus data was unzipped.

    """
    size_limit: int = field(default=10**9)
    """Limit on byte size of XML file, which otherwise tanks the process because
    of memeory constraints.

    """
    @property
    def file_paths(self) -> Iterable[Path]:
        parts: List[str] = Path(self.file_glob).parts
        root: Path = Path(parts[0])
        logger.info(f'iterating over root={root}, glob={"/".join(parts)}')
        glob: str = str(Path(*parts[1:]))
        files: Iterable[Path] = root.glob(glob)
        files = filter(lambda p: os.stat(p).st_size < self.size_limit, files)
        return files

    def _parse_toks(self, sent: str) -> List[str]:
        return re.findall(self._TOK_REGEX, sent)

    def _parse_sents(self, text: str, sents: List[Sentence], six: int) -> int:
        para: str
        for para in filter(lambda s: len(s) > 0, text.split('\n')):
            for i in re.finditer(self._SENT_REGEX, para):
                sent: str = i.group(0)
                toks: List[str] = self._parse_toks(sent)
                sents.append(Sentence(tuple(toks), str(six)))
                six += 1
        return six

    @abstractmethod
    def _parse_docs(self) -> Iterable[Document]:
        pass

    def create(self) -> Iterable[Document]:
        """Generates an iterable sequence of parsed documents."""
        return filter(lambda d: d is not None, self._parse_docs())


@dataclass
class XmlDocumentFactory(DocumentFactory):
    """Parses the corpus XML files.

    """
    def _parse_tok_doc(self, xr: XmlResource):
        sents: List[Sentence] = []
        for sent_elem in xr.xpath('//s'):
            toks: Tuple[str] = tuple(map(str, sent_elem.xpath('w/text()')))
            if len(toks) > 0:
                sid: int = sent_elem.attrib.get('id')
                sents.append(Sentence(toks, sid))
        return Document(xr.path, tuple(sents))

    def _parse_sent_doc(self, xr: XmlResource):
        sents: List[Sentence] = []
        for sent_elem in xr.xpath('/text/s'):
            tns = sent_elem.xpath('./text()')
            if len(tns) > 0:
                snode = tns[0]
                sid: int = sent_elem.attrib.get('id')
                text: str = str(snode)
                sents.append(Sentence(text.split(), sid))
        return Document(xr.path, tuple(sents))

    def _parse_docs(self) -> Iterable[Document]:
        xf: Path
        for xf in self.file_paths:
            logger.info(f'processing: {xf}')
            xr = XmlResource(xf)
            doc: Document = None
            try:
                doc = self._parse_tok_doc(xr)
            except Exception as e:
                logger.warning(f'error while parsing: {e}')
            if doc is None or len(doc) > 0:
                yield doc


@dataclass
class WikipediaDocumentFactory(XmlDocumentFactory):
    def _parse_tok_doc(self, xr: XmlResource):
        def sub_template(template: wtp.Template) -> str:
            if len(template.arguments) > 0:
                return template.arguments[0].value
            return template.name

        sents: List[Sentence] = []
        six: int = 0
        for sent_elem in xr.xpath("//w:text[@bytes > 70]"):
            root = wtp.parse(sent_elem.text)
            sec: wtp.Section
            for sec in root.sections:
                text: str = wtp.remove_markup(
                    sec.contents, replace_templates=sub_template)
                six = self._parse_sents(text, sents, six)
        return Document(xr.path, tuple(sents))


@dataclass
class OscarDocumentFactory(DocumentFactory):
    _OK_WARNS: ClassVar[Set[str]] = set('tiny short_sentences'.split())
    _ALL_WARNS: ClassVar[Set[str]] = _OK_WARNS | set('header footer noisy'.split())

    min_lang_prob: float = field(default=0.9)

    def _parse_content(self, file_content: TextIOBase) -> Iterable[str]:
        min_lang_prob: float = self.min_lang_prob
        empty_set: Set[str] = set()
        line: str
        for line in map(str.strip, file_content.readlines()):
            jcont: Dict[str, Any] = json.loads(line)
            meta: Dict[str, Any] = jcont['metadata']
            qual_warn_lst: Set[str] = meta['quality_warnings']
            qual_warns: Set[str] = empty_set
            if qual_warn_lst is not None:
                qual_warns = set(qual_warn_lst)
            bad_warns: Set[str] = qual_warns - self._OK_WARNS
            unk_warns: Set[str] = bad_warns - self._ALL_WARNS
            if len(unk_warns) > 0:
                raise ValueError(f'Unkonwn errors: {unk_warns}')
            if len(bad_warns) > 0:
                continue
            sents: List[str] = jcont['content'].split('\n')
            sent_ids: List[Dict[str, Any]] = meta['sentence_identifications']
            if len(sents) != len(sent_ids):
                raise ValueError(f'read {len(sents)} sentences, ' +
                                 f'but got {len(sent_ids)} sentence IDs')
            sent: str
            for sent, sent_id in zip(sents, sent_ids):
                if sent_id is not None and \
                   sent_id['label'] == 'sq' and \
                   sent_id['prob'] > min_lang_prob:
                    #yield f"{sent} ({sent_id['prob']})"
                    yield sent

    def _parse_file(self, path: Path) -> Iterable[Document]:
        sents: List[Sentence] = []
        with open(path) as f:
            sent: str
            for six, sent in enumerate(self._parse_content(f)):
                sents.append(Sentence(tuple(self._parse_toks(sent)), six))
            yield Document(path, tuple(sents))

    def _parse_docs(self) -> Iterable[Document]:
        xf: Path
        for xf in self.file_paths:
            logger.info(f'processing: {xf}')
            doc: Document
            for doc in self._parse_file(xf):
                yield doc


@dataclass
class FeatureDocumentStash(ReadOnlyStash):
    """A stash that creates instances of
    :class:`~zensols.nlp.container.FeatureDocument` from data in the SQLite DB
    file.

    """
    doc_persister: DbPersister = field()
    """SQLite document persister."""

    sent_persister: DbPersister = field()
    """SQLite sentence persister."""

    max_sent_size: int = field()
    """The maximum sentence size per document when querying the SQLite DB.  If
    the document has more sentences than this value, it is broken up in to
    smaller documents with sentences having this size.

    """
    def _doc_from_db(self, doc_id: int) -> Iterable[Document]:
        def map_sent(row: Tuple) -> Sentence:
            sent = Sentence.from_span_str(*row)
            if sent.tokens[0] == '"' and sent.tokens[-1] == '"':
                sent.tokens = sent.tokens[1:-1]
            return sent

        def map_chunk(rows: Tuple[Tuple]) -> Document:
            sents: Iterable[Sentence] = map(map_sent, rows)
            return Document(path=None, sents=tuple(sents))

        sp: DbPersister = self.sent_persister
        rows: Tuple[Tuple] = sp.execute_by_name(
            'select_sent_by_doc_id', params=(doc_id,))
        return map(map_chunk, chunks(rows, self.max_sent_size))

    def load(self, doc_id: str) -> Iterable[FeatureDocument]:
        def map_doc(doc: Document) -> FeatureDocument:
            fdoc = Document.to_feature_document(doc)
            fdoc.corpus_id = doc_id
            return fdoc

        doc_id = int(doc_id)
        if self.exists(doc_id):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'getting doc: {doc_id}')
            docs: Iterable[Document] = self._doc_from_db(doc_id)
            return map(map_doc, docs)

    def keys(self) -> Iterable[str]:
        return self.doc_persister.get_keys()

    def exists(self, doc_id: str) -> bool:
        return self.doc_persister.exists(int(doc_id))


@dataclass
class CorpusCompiler(object):
    """Parse the Albanian corpus.

    """
    doc_factory: DocumentFactory = field()
    """Parses the corpus XML files and returns them as :class:`.Document` data
    instances.

    """
    lang: str = field()
    """The default language.  If not provided, :mod:`langdetect` is used."""

    sources_file: Path = field()
    """The corpus name to URL."""

    doc_stash: Stash = field()
    """Pickled :class:`.Document` instances of already parsed documents."""

    doc_persister: DbPersister = field()
    """SQLite document persister."""

    sent_persister: DbPersister = field()
    """SQLite sentence persister."""

    corpus_source_persister: DbPersister = field()
    """SQLite corpus source persister."""

    recreate: bool = field()
    """Whether to drop the DB before loading."""

    def parse_docs(self):
        """Parse the XML corpus files in to intermediate (temporary) pickled
        data files.

        """
        logger.info('parsing docs')
        self.doc_stash.clear()
        for did, doc in zip(it.count(), self.doc_factory.create()):
            self.doc_stash.dump(did, doc)

    @property
    def stats_df(self) -> pd.DataFrame:
        """A dataframe of statistcs of the corpus."""
        dp: DbPersister = self.doc_persister
        rows: List[Tuple[str, Any]] = []
        for col, qname in (('corpus sources', 'corp_doc_name_count'),
                           ('documents', 'doc_count'),
                           ('sentences', 'sent_count')):
            val: Any = dp.execute_singleton_by_name(qname)[0]
            rows.append((col, val))
        return pd.DataFrame(rows, columns='description value'.split())

    def _get_corpus_source_dataframe(self) -> pd.DataFrame:
        """Return a data frame with the following columns::

          * ``doc_id``: the ``doc_id`` (SQLite row ID) from the ``doc`` table

          * ``sid``: the ``sq.zip.<ID>`` parsed from the original file
            (i.e. ``17``).

          * ``name``: the name of the corpus (i.e. ``OpenSubtitles``)

          * ``file``: the corpus relative XML path from the path from the doc
            table

        """
        poffset: int = self.doc_factory.corpus_path_offset
        dp: DbPersister = self.doc_persister
        cols: List[str] = 'doc_id sid name file'.split()
        rows: List[Tuple[str, int, str, str]] = []
        did: int
        dir_id: str
        path_str: str
        for did, dir_id, path_str in dp.get():
            path = Path(path_str)
            parts: List[str] = path.parts
            sid: int = int(parts[poffset - 1])
            name: str = parts[poffset]
            sub_path: str = '/'.join(parts[poffset + 1:])
            rows.append((did, sid, name, sub_path))
        return pd.DataFrame(rows, columns=cols)

    def update_corpus_source(self):
        """Synchronize the corpus sources in the ``corp_doc`` table found in all
        documents.

        """
        df: pd.DataFrame = self._get_corpus_source_dataframe()
        rows = df.itertuples(index=False, name=None)
        self.corpus_source_persister.execute_no_read('delete_corp_doc')
        self.corpus_source_persister.insert_rows(rows)
        logger.info('finished corpus source sync')

    def update_corpus_url(self):
        """Update the ``corp_src`` corpus source table."""
        sp: DbPersister = self.corpus_source_persister
        df: pd.DataFrame = pd.read_csv(self.sources_file)
        self.corpus_source_persister.execute_no_read('delete_corp_src')
        for row in df.itertuples(index=False, name=None):
            sp.execute_no_read('insert_source', params=row)

    def _load_db(self):
        """Populate the SQLite DB from the pickled document stash.  This also
        calls :meth:`update_corpus_source` to add the document sources.

        """
        sp: DbPersister = self.sent_persister
        if self.recreate:
            sp.conn_manager.drop()
            sp.conn_manager.create()
        logger.info('loading db')
        for k, doc in self.doc_stash:
            if not self.recreate:
                mid = sp.execute('select max(cast(dir_id as decimal)) from doc')
                mid = mid[0][0]
                k = str(mid + 1)
            doc_id: int = sp.execute_no_read(
                'insert_doc', params=(k, str(doc.path)))
            sent_texts: List[Tuple] = []
            sent: Sentence
            for six, sent in enumerate(doc.sents):
                lang: str = None
                sid: str = sent.sid
                text: str = sent.text
                span_str: str = sent.span_str
                tok: str
                lang: str = self.lang
                if lang is None:
                    try:
                        lang = ld.detect(sent.text)
                    except Exception as e:
                        logger.warning(f'no language for: <{sent.text}>: {e}')
                # paranoia
                comp = Sentence.from_span_str(text, span_str, sid)
                if comp != sent:
                    raise ValueError(
                        f'Sentence reconstruction failed: {sent} != {comp}')
                sent_texts.append((doc_id, six, sid, lang, text, span_str, len(comp.tokens)))
            sp.insert_rows(sent_texts)
        logger.info(f'finished db load of {sp.conn_manager.db_file}')
        self.update_corpus_source()
        self.update_corpus_url()

    def compile(self, force: bool = False):
        """Compile the corpus in to an SQLite file.

        :param force: force recompilation of database (not doc parsing)

        """
        cnt: int = self.sent_persister.get_count()
        if force or cnt == 0:
            if len(self.doc_stash) == 0:
                self.parse_docs()
            self._load_db()
        else:
            logger.info(f'nothing needs loading: found {cnt} sentences')

    def _write_doc_tokenization(self, doc: FeatureDocument):
        from zensols.deeplearn.vectorize import FeatureVectorizerManager
        from zensols.deepnlp.transformer import TransformerFeatureVectorizer
        mng: FeatureVectorizerManager = \
            self.config_factory('language_vectorizer_manager')
        vec: TransformerFeatureVectorizer = mng['transformer_trainable']
        tdoc = vec.tokenize(doc)
        tdoc.write()


@dataclass
class CorpusDumper(object):
    """Dumps the cleaned corpus in memory or on the disk as CSV files.

    :deprecated: further sentence deletion and massaging was necessary after
                 this class was written

    :see: :class:`.DatasetManager`

    """
    feat_doc_stash: Stash = field()
    """A stash the generates :class:`~zensols.nlp.FeatureDocument` instances
    from the database from :class:`.FeatureDocumentStash`.

    """
    row_size: int = field()
    """The row size of each (except the last) :class:`pd.DataFrame` when
    iterating.  This is also the row size of the output CSV file.

    """
    def get_sentences(self) -> Iterable[Tuple[int, FeatureSentence]]:
        """Return an iterator that traverses through all sentences in the
        corpus.  A tuple with corresponding document IDs to which they belong is
        also provided.

        :return: tuples of form ``(<document ID>, <sentence>)``

        """
        stash: Stash = self.feat_doc_stash
        keys: List[str] = list(stash.keys())
        random.shuffle(keys)
        did: str
        for did in keys:
            docs: Iterable[FeatureDocument] = stash[did]
            for doc in docs:
                sent: FeatureSentence
                for sent in doc.sents:
                    yield (did, sent)

    def get_dataframes(self) -> Iterable[pd.DataFrame]:
        """Return data frames of corpus sentences.  The length of each is given
        by :obj:`row_size`.

        """
        def map_chunk(rows: Tuple[Tuple[int, FeatureSentence]]) -> pd.DataFrame:
            rows = map(lambda r: (r[0], r[1].corpus_id, r[1].text), rows)
            return pd.DataFrame(rows, columns='did sid text'.split())

        rows_iter: Iterable[Tuple[int, str, str]] = self.get_sentences()
        return map(map_chunk, chunks(rows_iter, self.row_size))

    def dump(self, output_dir: Path, dataframe_limit: int = sys.maxsize):
        """Dump Pandas data frames as CSV files.

        :param output_dir: the directory to generate the files, which is
                           generated if it doesn't exist already

        :param dataframe_limit: the maximum number of files to write

        :see: :meth:`get_dataframes`

        """
        output_dir.mkdir(parents=True, exist_ok=True)
        dfs: Iterable[pd.DataFrame] = self.get_dataframes()
        for i, df in it.islice(zip(it.count(), dfs), dataframe_limit):
            path = output_dir / f'{i}.csv'
            df.to_csv(path)
            logger.info(f'wrote: {path}')


@dataclass
class Corpus(object):
    """Reads and writes the arrow Dataset used by the :class:`.Trainer`.

    """
    persister: DbPersister = field()
    dataset_path: Path = field()
    batch_size: int = field()
    test_size: float = field()
    sent_limit: int = field(default=sys.maxsize)
    read_limit: int = field(default=None)

    def write_dataset(self):
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([pa.field('text', pa.string())])
        with cursor(self.persister, name='select_sents',
                    params=(self.sent_limit,)) as c:
            with pa.OSFile(str(self.dataset_path), 'wb') as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    while True:
                        sents: Iterable[str] = tuple(it.islice(
                            map(lambda r: r[0], c), self.batch_size))
                        if len(sents) == 0:
                            break
                        batch = pa.record_batch(
                            [pa.array(sents, type=pa.string())], schema)
                        writer.write(batch)

    @property
    def table(self) -> Table:
        with pa.OSFile(str(self.dataset_path), 'rb') as source:
            table: Table = pa.ipc.open_file(source).read_all()
        if self.read_limit is not None:
            table = table.slice(length=self.read_limit)
        return table

    @property
    def tokenizer_dataset(self) -> Dataset:
        return Dataset(self.table)

    @property
    def model_dataset(self) -> Dataset:
        table: Table = self.table
        ds = Dataset(table)
        return ds.train_test_split(test_size=self.test_size, seed=0)

    @property
    def vocab(self) -> Set[str]:
        toks: Tuple[str] = self.persister.execute_by_name(
            name='select_vocab',
            map_fn=lambda r: r[0])
        return set(toks)
