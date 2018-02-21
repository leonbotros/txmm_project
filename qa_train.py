import json

import numpy as np
from keras import optimizers
from keras.layers import Embedding, Bidirectional, TimeDistributed
from keras.layers import Input, Dense, Flatten, Concatenate, Activation
from keras.models import Model
from keras.layers import LSTM
# from keras.layers import GRU as LSTM # uncomment to use GRUs instead of LSTMs
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from attention import Attention

embedding_dim = 100 # embedding dim
max_context_seq_length = 350 # max context sequence length
max_question_seq_length = 50  # max question sequence length
units = 128  # number of LSTM/GRU units

n_samples = None  # None means all
epochs = 5
learning_rate = 0.01
# decay = 0.001
batch_size = 32
val_split = 0.1

def get_samples(train_data):
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


def load_glove_model(file):
    with open(file, 'r', encoding='utf8') as f:
        return {line.split()[0]: np.asarray(line.split()[1:], dtype='float32') for line in f}

def get_embedding_matrix(word_index, glove_model):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = glove_model[word]
        except KeyError:
            pass  # leave it zeros if the word is not found
    return embedding_matrix


def find_in_padded_seq(answer_seq, context_padded_seq):
    for i in range(len(context_padded_seq)):
        if np.array_equal(context_padded_seq[i: i + len(answer_seq)], answer_seq):
            if i + len(answer_seq) >= max_context_seq_length:
                return -1, 0  # the answer was cut out due to our max sequence length
            else:
                return i, i + len(answer_seq)
    return -1, 0  # answer was not found somehow

def get_best_answer_span(start_probs, end_probs):
    """ Finds best answer span (i, j) s.t.
     i <= j and start_probs[i] * end_probs[j] is maximum
     NOTE: very very very inefficient solution
    """
    sorted_spans = sorted([(i, j, k*l) for i, k in enumerate(start_probs) for j, l in enumerate(end_probs) if i <= j], key=lambda x: x[2])
    return sorted_spans[0][0], sorted_spans[0][1]

def get_model(embedding_matrix):
    context_embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                        output_dim=embedding_matrix.shape[1],
                                        weights=[embedding_matrix],
                                        input_length=max_context_seq_length,
                                        trainable=False)

    question_embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                         output_dim=embedding_matrix.shape[1],
                                         weights=[embedding_matrix],
                                         input_length=max_question_seq_length,
                                         trainable=False)

    # The 2 inputs to our model
    c_seq_input = Input(shape=(max_context_seq_length,), dtype='int32')  # [batch_size, n]
    q_seq_input = Input(shape=(max_question_seq_length,), dtype='int32')  # [batch_size, m]

    # Embed both question and context
    c_embedded = context_embedding_layer(c_seq_input)  # [batch_size, n, d]
    q_embedded = question_embedding_layer(q_seq_input)  # [batch_size, m, d]

    # Bidirectional LSTMs/GRUs used as encoders
    c_encoder_out = Bidirectional(LSTM(units, return_sequences=True))(c_embedded)  # [batch_size, n, 2l]

    q_encoder_out = Bidirectional(LSTM(units, return_sequences=True))(q_embedded)  # [batch_size, n, 2l]

    # Interaction/attention layer, output shape
    G = Attention()([c_encoder_out, q_encoder_out])  # [batch_size, n, 4l]

    m_1 = Bidirectional(LSTM(units, return_sequences=True))(G)  # [batch_size, n, 2l]

    concat1_out = Concatenate(axis=-1)([G, m_1])

    ps_start_ = TimeDistributed(Dense(1))(concat1_out)  # [batch_size, n, 1]
    ps_start_flatten = Flatten()(ps_start_)  # [batch_size, n]
    ps_start = Activation('softmax')(ps_start_flatten)
    # ps_start = Dense(max_context_seq_length, activation='softmax')(ps_start_flatten)  # [batch_size, n]

    # m_2 = Bidirectional(LSTM(units, return_sequences=True))(m_1)  # [batch_size, n, 2l]
    # concat2_out = Concatenate(axis=-1)([G, m_2])  # [batch_size, n, 4l]

    ps_end_ = TimeDistributed(Dense(1))(concat1_out)  # [batch_size, n, 1]
    ps_end_flatten = Flatten()(ps_end_)  # [batch_size, n]
    ps_end = Activation('softmax')(ps_end_flatten)
    # ps_end = Dense(max_context_seq_length, activation='softmax')(ps_end_flatten)  # [batch_size, n]

    model = Model(inputs=[c_seq_input, q_seq_input], outputs=[ps_start, ps_end])
    return model


def main():
    train_data = json.load(open("data/train-v1.1.json"))['data']
    samples = get_samples(train_data)[:n_samples]
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

    print("Locating answer indexes in paddes context sequences")
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

    # check if nothing went wrong with preprocessing network inputs
    # inv_word_index = {i: w for w, i in word_index.items()}
    # i = 500
    # print([inv_word_index[x] for  x in context_seqs_padded[i] if x])
    # print([inv_word_index[x] for x in question_seqs_padded[i] if x])
    # print([inv_word_index[x] for x in context_seqs_padded[i][start[i]: end[i]] if x])

    print(context_seqs_padded.shape, question_seqs_padded.shape, a_s_y.shape, a_e_y.shape)
    embedding_matrix = get_embedding_matrix(word_index, glove_model)
    print(embedding_matrix.shape)

    model = get_model(embedding_matrix)
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.summary()

    model.fit([context_seqs_padded, question_seqs_padded],
              [a_s_y, a_e_y],
              epochs=epochs,
              batch_size=batch_size,
              validation_split=val_split)

    model.save('simple_bidaf2.h5')

    ps_start, ps_end = model.predict([context_seqs_padded, question_seqs_padded])
    for s, e in zip(ps_start, ps_end):
        max_s = np.argmax(s)
        max_e = np.argmax(e)
        print(max_s, max_e)
        print(get_best_answer_span(s, e))


if __name__ == "__main__":
    main()
