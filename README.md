# TxMM Project: Question answering using neural networks

SQuAD data goes in the /data directory.
Glove embeddings go in the /glove directory.

# Links

[train set link](bhttps://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
[dev set link](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
[glove embeddings link](http://nlp.stanford.edu/data/glove.6B.zip)

# Training

Run qa_train.py

# Testing on dev set

Run `qa_predict.py`

# Evaluating

Run `squad/evaluate1.1.py data/dev-v1.1json predictions.txt`


