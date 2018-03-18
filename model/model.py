import numpy as np
from keras.layers import Embedding, Bidirectional, TimeDistributed
from keras.layers import Input, Dense, Flatten, Concatenate, Activation
from keras.layers import LSTM
from keras.models import Model

from model.attention import Attention

# from keras.layers import GRU as LSTM # uncomment to use GRUs instead of LSTMs


embedding_dim = 100  # embedding dim
max_context_seq_length = 350  # max context sequence length
max_question_seq_length = 50  # max question sequence length
units = 128  # number of LSTM/GRU units
dropout = 0.2

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


def get_model(embedding_matrix, name):
    context_embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                        output_dim=embedding_matrix.shape[1],
                                        weights=[embedding_matrix],
                                        input_length=max_context_seq_length,
                                        trainable=False,
                                        name='c_emb' + name)

    question_embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                         output_dim=embedding_matrix.shape[1],
                                         weights=[embedding_matrix],
                                         input_length=max_question_seq_length,
                                         trainable=False,
                                         name='q_emb' + name)

    # The 2 inputs to our model
    c_seq_input = Input(shape=(max_context_seq_length,), dtype='int32')  # [batch_size, n]
    q_seq_input = Input(shape=(max_question_seq_length,), dtype='int32')  # [batch_size, m]

    # Embed both question and context
    c_embedded = context_embedding_layer(c_seq_input)  # [batch_size, n, d]
    q_embedded = question_embedding_layer(q_seq_input)  # [batch_size, m, d]

    # Bidirectional LSTMs/GRUs used as encoders
    c_encoder_out = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(c_embedded)  # [batch_size, n, 2l]

    q_encoder_out = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(q_embedded)  # [batch_size, n, 2l]

    # Interaction/attention layer, output shape
    G = Attention()([c_encoder_out, q_encoder_out])  # [batch_size, n, 4l]

    # Modeling layer
    m_1 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(G)  # [batch_size, n, 2l]
    m_2 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(m_1)  # [batch_size, n, 2l]
    m_3 = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(m_2)

    concat1_out = Concatenate(axis=-1)([G, m_2])

    ps_start_ = TimeDistributed(Dense(1))(concat1_out)  # [batch_size, n, 1]
    ps_start_flatten = Flatten()(ps_start_)  # [batch_size, n]
    ps_start = Activation('softmax')(ps_start_flatten)

    concat2_out = Concatenate(axis=-1)([G, m_3])
    ps_end_ = TimeDistributed(Dense(1))(concat2_out)  # [batch_size, n, 1]
    ps_end_flatten = Flatten()(ps_end_)  # [batch_size, n]
    ps_end = Activation('softmax')(ps_end_flatten)

    model = Model(inputs=[c_seq_input, q_seq_input], outputs=[ps_start, ps_end])
    return model
