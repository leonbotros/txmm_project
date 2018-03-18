# TxMM Project: Question answering using neural networks

# Links

[Train set link](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)

[Dev set link](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

[Glove embeddings link](http://nlp.stanford.edu/data/glove.6B.zip)

SQuAD data goes in the data/ directory.
Glove embeddings go in the glove/ directory.

# Training

To run 10 epochs with a batch size of 64 and an initial learning rate of 0.1 run:

`qa_train.py --e 10 -b 64 -lr 0.5`

# Testing on dev set

To use the learned weights to predict on the dev set run:

`qa_predict.py -w weights/whatever.h5 -p txmm_project/predictions.json`

# Evaluating

To evaluate the predictions using the official SQuAD evaluate script run:

`squad/evaluate-v1.1.py predictions.json`