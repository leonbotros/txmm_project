import json

import numpy as np
from keras import optimizers
# from keras.layers import GRU as LSTM # uncomment to use GRUs instead of LSTMs
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from model.model import get_model, load_glove_model, get_embedding_matrix

embedding_dim = 100  # embedding dim
max_context_seq_length = 350  # max context sequence length
max_question_seq_length = 50  # max question sequence length
units = 128  # number of LSTM/GRU units

n_samples = None  # None means all
epochs = 5
learning_rate = 0.01
# decay = 0.001
batch_size = 32
val_split = 0.1


def get_train_samples(train_data):
    samples = []
    for subj in train_data:
        paragraphs = subj['paragraphs']
        for p in paragraphs:
            context = p['context']
            for q in p['qas']:
                start = q['answers'][0]['answer_start']
                end = start + len(q['answers'][0]['text'])
                samples.append((context, q['question'], (start, end)))
    return samples


def find_in_padded_seq(answer_seq, context_padded_seq):
    for i in range(len(context_padded_seq)):
        if np.array_equal(context_padded_seq[i: i + len(answer_seq)], answer_seq):
            if i + len(answer_seq) >= max_context_seq_length:
                return -1, 0  # the answer was cut out due to our max sequence length
            else:
                return i, i + len(answer_seq)
    return -1, 0  # answer was not found somehow


def main():
    train_data = json.load(open("data/train-v1.1.json"))['data']
    samples = get_train_samples(train_data)[:n_samples]
    print('Training samples: %d' % len(samples))

    assert embedding_dim in [50, 100, 200, 300]
    glove_path = 'glove/glove.6B.%dd.txt' % embedding_dim
    print('Loading glove model')
    glove_model = load_glove_model(glove_path)
    print('Done loading glove model')

    contexts, questions, answers = zip(*samples)

    # Scan every word in the questions and contexts and index them
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(contexts + questions)
    word_index = tokenizer.word_index
    print('Done fitting tokenizer on texts, found %d unique tokens' % len(word_index))

    print("Tokenizing questions and contexts")
    context_seqs = tokenizer.texts_to_sequences(contexts)
    question_seqs = tokenizer.texts_to_sequences(questions)

    # find start and end location in tokenized representation
    answers_str = [context[s: e] for context, (s, e) in zip(contexts, answers)]
    answer_seqs = tokenizer.texts_to_sequences(answers_str)

    # Pad the question- and context sequences to the same length
    context_seqs_padded = pad_sequences(context_seqs, maxlen=max_context_seq_length, padding='post')
    question_seqs_padded = pad_sequences(question_seqs, maxlen=max_question_seq_length, padding='post')

    print('Longest sequences:\n context: %d \n question: %d' % (max([len(s) for s in context_seqs]),
                                                                max([len(s) for s in question_seqs])))

    c_proportion = float(len([c for c in context_seqs if len(c) <= max_context_seq_length])) / len(context_seqs)
    q_proportion = float(len([q for q in question_seqs if len(q) <= max_question_seq_length])) / len(question_seqs)

    print('Proportion of contexts smaller or equal to %d: %f\nProportion questions smaller or equal to %d: %f' % (
        max_context_seq_length, c_proportion, max_question_seq_length, q_proportion))

    print("Locating answer indexes in padded context sequences")
    ans_in_context = [find_in_padded_seq(np.asarray(answer_seq), context_seq) for answer_seq, context_seq in
                      zip(answer_seqs, context_seqs_padded)]

    start, end = zip(*ans_in_context)

    # remove questions, contexts, answer triplets that have no located answer in our tokenized sequence representation
    to_remove = [i for i, s in enumerate(start) if s == -1]
    print('Removing %d samples' % len(to_remove))
    context_seqs_padded = np.delete(context_seqs_padded, to_remove, axis=0)
    question_seqs_padded = np.delete(question_seqs_padded, to_remove, axis=0)
    start = np.delete(start, to_remove)
    end = np.delete(end, to_remove)
    # categorical labels of floats
    a_s_y = to_categorical(np.asarray(start, dtype='float32'), num_classes=max_context_seq_length)
    a_e_y = to_categorical(np.asarray(end, dtype='float32'), num_classes=max_context_seq_length)

    print(context_seqs_padded.shape, question_seqs_padded.shape, a_s_y.shape, a_e_y.shape)
    embedding_matrix = get_embedding_matrix(word_index, glove_model)
    print(embedding_matrix.shape)

    model = get_model(embedding_matrix, name='train')

    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.summary()

    model.fit([context_seqs_padded, question_seqs_padded],
              [a_s_y, a_e_y],
              epochs=epochs,
              batch_size=batch_size,
              validation_split=val_split)

    model.save_weights('simple_bidaf.h5')


if __name__ == "__main__":
    main()
