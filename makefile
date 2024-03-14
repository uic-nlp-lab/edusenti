#@meta {desc: 'EduSenti build automation', date: '2024-03-06'}


MTARG =		target
ENTRY = 	./harness.py


# install dependencies
.PHONY:		deps
deps:
		pip install -r src/requirements.txt

# vectorize the batches
.PHONY:		batch
batch:
		$(ENTRY) batch --override edusenti_default.lang=en
		$(ENTRY) batch --override edusenti_default.lang=sq

# compile the corpus in to an SQLite file
.PHONY:		compcomp
compcomp:
		$(ENTRY) fineclean
		mkdir -p data/sq/ft
		$(ENTRY) finedscomp


# build and fine-tune the sentiment corpus
.PHONY:		finetune
finetune:	corpcomp
		$(ENTRY) config --cnffmt text > data/sq/ft/config.txt
		$(ENTRY) config --cnffmt json > data/sq/ft/config.json
		( nohup $(ENTRY) finetrainmodel > train.log 2>&1 ; \
			mv train.log data/sq/ft/ ) &

# clean create/derived files
.PHONY:		clean
clean:
		rm -fr $(MTARG)
		find . -type d -name __pycache__ -prune -exec rm -r {} \;

# remove the fine-tune scorpus
.PHONY:		cleanfinetune
cleanfinetune:
		rm -r data/sq/ft


.PHONY:		cleanall
cleanall:	clean
		$(ENTRY) clean --clevel 2
