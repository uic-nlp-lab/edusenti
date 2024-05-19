# EduSenti: Education Review Sentiment in Albanian

This repo contains the code, data, and instructions to reproduce the results in
the paper [RoBERTa Low Resource Fine Tuning for Sentiment Analysis in
Albanian].


## Inclusion in Your Projects

If you want to use sentiment model in your own work, follow the instructions
using the [zensols.edusenti] repository.  If you use our model or API, please
[cite](#citation) our paper.


## Reproducing the Results

The source code used Python 3.9.9 using the CUDA 11 drivers.  To reproduce the
results:
1. Clone the paper repo: `git clone https://github.com/uic-nlp-lab/edusenti`
1. Go into it and prepare the corpus: `cd edusent ; mkdir -p corpus/finetune`
1. Download and extract the [Pretraining Corpus](#albanian-pretraining-corpus):
   `wget -O - https://zenodo.org/records/10778230/files/albanian-sq.sqlite3.bz2 | bzip2 -cd > corpus/finetune/sq.sqlite3`
1. Install dependencies: `pip install --use-deprecated=legacy-resolver -r src/requirements.txt`
1. Confirm the fine-tune sentiment corpus is readable: `./harness.py finestats`
1. Vectorize English sentiment corpus batches: `./harness.py batch --override
   edusenti_default.lang=en`
1. Vectorize English sentiment corpus batches: `./harness.py batch --override
   edusenti_default.lang=sq`
1. Train and test the Albanian model on GloVE 50D embeddings:
   `./harness.py traintest`
1. Train and test the English model:
   `./harness.py traintest --override edusenti_default.lang=en`

Use the [Jupyter Notebook](notebook/edusenti.ipynb) to train all the variations
(and [configurations](./models)) of the model and print the results.

Note that the repository has a lot of commands and code for creating the
[Pretraining Corpus](#albanian-pretraining-corpus).  However, those steps can
be skipped with the `wget` download command above.

**Important**: The focus on this work was Albanian and English was only used
for comparison.  For this reason, the attention was on Albanian for
reproduction of results and not English, which is why the English sentiment
dataset splits were not recorded.


## Albanian Sentiment Corpus

Both the Albanian (sq) and English (en) EduSenti corpus are available [in this
file](corpus/edusenti-corpus.zip).


## Albanian Pretraining Corpus

The [Albanian pretraining corpus] used for pertaining large language models is
an SQLite (v3) database with the following tables:

* `corp_src`: the sources of the Albanian text
* `corp_doc`: the corpus source (names) and source files
* `doc`: joins from sentences to corpus document source (`corp_doc`)
* `sent`: the Albanian sentences with tokenization and token length

This query shows how to get the corpus sources and constituent counts:
```sql
select cs.id as name, cs.url, count(*) as count
  from corp_src as cs, corp_doc as cd, doc as d, sent as s
  where cd.name = cs.id and
        cd.doc_id = d.rowid and
	cd.doc_id = s.doc_id
  group by cs.id;
```

See the [corpus creation SQL](resources/finetune.sql) for useful queries and to
see how it was procured/cleaned.


## Citation

If you use this project in your research please use the following BibTeX entry:

```bibtex
@inproceedings{nuci-etal-2024-roberta-low,
    title = "{R}o{BERT}a Low Resource Fine Tuning for Sentiment Analysis in {A}lbanian",
    author = "Nuci, Krenare Pireva  and
      Landes, Paul  and
      Di Eugenio, Barbara",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1233",
    pages = "14146--14151"
}
```


## License

[MIT License]

Copyright (c) 2024 Paul Landes and Krenare Pireva Nuci


<!-- links -->

[MIT License]: https://opensource.org/licenses/MIT
[Albanian pretraining corpus]: https://zenodo.org/records/10778230
[zensols.edusenti]: https://github.com/plandes/edusenti
[RoBERTa Low Resource Fine Tuning for Sentiment Analysis in Albanian]: https://aclanthology.org/2024.lrec-main.1233
