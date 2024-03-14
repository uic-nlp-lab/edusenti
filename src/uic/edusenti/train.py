"""Taken from the `HF tutorial`_.

:see: `HF tutorial <https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt>`_

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
import logging
import collections
import numpy as np
from pathlib import Path
import platform
from datasets import Dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, BatchEncoding,
    DataCollatorForLanguageModeling, default_data_collator,
    PreTrainedTokenizerFast, TrainingArguments
)
from transformers import Trainer as HuggingFaceTrainer
from zensols.util import APIError
from zensols.persist import persisted
from . import Corpus

logger = logging.getLogger(__name__)


@dataclass
class Trainer(object):
    """A base class for the tokenizer and model trainers.

    """
    output_dir: Path = field()
    """The directory where the tokenizer and model files are saved."""

    corpus: Corpus = field()
    """The manager that reads (and writes) the dataset from the file system."""

    hub_model_id: str = field()
    """The HuggingFace ID used for the distributed model."""

    model_checkpoint_model_id: str = field()
    """The huggingface unique model ID."""

    tokenizer_checkpoint_model_id: str = field()
    """The huggingface unique tokenizer ID."""

    def __post_init__(self):
        from zensols.deepnlp import transformer
        transformer.suppress_warnings()
        if self.tokenizer_checkpoint_model_id is None:
            self.tokenizer_checkpoint_model_id = self.model_checkpoint_model_id

    @property
    def has_checkpoint_tokenizer(self) -> bool:
        """Whether a trained checkpoint tokenizer has been provided (per the
        configuration).

        """
        return self.tokenizer_checkpoint_model_id != \
            self.model_checkpoint_model_id

    @property
    @persisted('_tokenizer')
    def tokenizer(self) -> AutoTokenizer:
        """The checkpoint tokenizer."""
        logger.info(
            f'loading tokenizer from {self.tokenizer_checkpoint_model_id}')
        return AutoTokenizer.from_pretrained(
            self.tokenizer_checkpoint_model_id)

    @property
    @persisted('_model')
    def model(self) -> AutoModelForMaskedLM:
        """The checkpoint model."""
        logger.info(f'loading model from {self.model_checkpoint_model_id}')
        return AutoModelForMaskedLM.from_pretrained(
            self.model_checkpoint_model_id)


@dataclass
class TokenizerTrainer(Trainer):
    """Creates a tokenizer data model.

    """
    batch_size: int = field(default=1000)
    """The batch size used for batch iteration."""

    add_portion: float = field(default=0.5)
    """The additional number of tokens in addition to the model checkpoint's
    tokenizer vocab size.

    """
    def train(self) -> PreTrainedTokenizerFast:
        def batch_iterator():
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i + batch_size]['text']

        logger.info('training tokenizer')
        if not self.has_checkpoint_tokenizer:
            raise APIError('No check point tokenizer provided')
        batch_size: int = self.batch_size
        dataset: Dataset = self.corpus.tokenizer_dataset
        prev_tokenizer: PreTrainedTokenizerFast = self.tokenizer
        prev_vocab: int = len(prev_tokenizer.vocab)
        add_vocab: int = int(prev_vocab * self.add_portion)
        tot_vocab: int = add_vocab + prev_vocab
        if logger.isEnabledFor(logging.INFO):
            logger.info('retraining tokenizer with vocab: ' +
                        f'{prev_vocab} -> {tot_vocab}')
        tokenizer: PreTrainedTokenizerFast = prev_tokenizer.\
            train_new_from_iterator(batch_iterator(), tot_vocab)

        tokenizer.save_pretrained(self.output_dir)
        logger.info(f'saved model to {self.output_dir}')

        return tokenizer


@dataclass
class MaskedModelTrainer(Trainer):
    """A masked language model trainer.

    """
    chunk_size: int = field()
    """The grouped dataset size chunk size."""

    use_whole_word_masking: bool = field()
    """Whether to use whole word rather than word piece masking."""

    mask_probability: float = field()
    """Whole masking probability."""

    batch_size: int = field()
    """The size of each batch to train on."""

    trainer_args: Dict[str, Any] = field()
    """The arguments given to the :class:`.HuggingFaceTrainer`."""

    def _tokenize_function(self, examples: Dict[str, Any]) -> BatchEncoding:
        result = self.tokenizer(examples['text'])
        if self.tokenizer.is_fast:
            result['word_ids'] = [result.word_ids(i)
                                  for i in range(len(result['input_ids']))]
        return result

    def _get_dataset(self) -> Tuple[Dataset, Dataset]:
        text_dataset: Dataset = self.corpus.model_dataset
        tokenized_dataset = text_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text'])
        return text_dataset, tokenized_dataset

    def _group_texts(self, examples: LazyBatch):
        chunk_size: int = self.chunk_size
        # concatenate all texts
        concatenated_examples = {k: sum(examples[k], [])
                                 for k in examples.keys()}
        # compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # we drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size]
                for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # create a new labels column
        result['labels'] = result['input_ids'].copy()
        return result

    def _get_grouped_dataset(self) -> DatasetDict:
        text_dataset, tokenized_dataset = self._get_dataset()
        return tokenized_dataset.map(self._group_texts, batched=True)

    def _whole_word_masking_data_collator(self, features: Dict[str, Any]):
        for feature in features:
            word_ids = feature.pop('word_ids')

            # create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # randomly mask words
            mask = np.random.binomial(1, self.mask_probability, (len(mapping),))
            input_ids = feature['input_ids']
            labels = feature['labels']
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = self.tokenizer.mask_token_id
            feature['labels'] = new_labels

        return default_data_collator(features)

    def _default_masking_data_collator(self, features: Dict[str, Any]):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mask_probability)
        for feature in features:
            feature.pop('word_ids')
        return data_collator(features)

    def train(self):
        logger.info('training model')
        dataset: DatasetDict = self._get_grouped_dataset()
        # show the training loss with every epoch
        logging_steps: int = len(dataset['train']) // self.batch_size
        if self.use_whole_word_masking:
            data_collator: Callable = self._whole_word_masking_data_collator
        else:
            data_collator: Callable = self._default_masking_data_collator
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f'training model: {self.hub_model_id} to {self.output_dir} ' +
                f'using tokenizer {self.tokenizer_checkpoint_model_id}')
        tokenizer: AutoTokenizer = self.tokenizer
        model: AutoModelForMaskedLM = self.model
        if self.has_checkpoint_tokenizer:
            model.resize_token_embeddings(len(tokenizer))
        training_args = TrainingArguments(
            overwrite_output_dir=True,
            push_to_hub=False,
            output_dir=Path(self.output_dir),
            hub_model_id=self.hub_model_id,
            evaluation_strategy='epoch',
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=not platform.system() == 'Darwin',
            logging_steps=logging_steps,
            remove_unused_columns=False,
            save_strategy='epoch',
            **self.trainer_args)
        trainer = HuggingFaceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
            tokenizer=tokenizer)
        trainer.train()
