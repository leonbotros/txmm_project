import json

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from model.model import *


def get_dev_samples(dev_data):
    samples = []
    for subj in dev_data:
        paragraphs = subj['paragraphs']
        title = subj['title']
        for p in paragraphs:
            context = p['context']
            for q in p['qas']:
                id = q['id']
                question = q['question']
                answers = q['answers']
                samples.append((id, context, question))
    return samples


def get_best_answer_span(start_probs, end_probs):
    """ Finds best answer span (i, j) s.t.
     i <= j and start_probs[i] * end_probs[j] is maximum
    """
    max_i, max_j = 0, 1
    max_prob = start_probs[max_i] * end_probs[max_j]

    for j, _ in enumerate(start_probs):
        if start_probs[max_i] * end_probs[j] > max_prob:
            max_j = j
            max_prob = start_probs[max_i] * end_probs[j]
        if start_probs[j] > start_probs[max_i]:
            max_i = j

    return max_i, max_j


def main():
    dev_data = json.load(open("data/dev-v1.1.json"))['data']
    samples = get_dev_samples(dev_data)[:]
    ids, contexts, questions = zip(*samples)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(contexts + questions)
    word_index = tokenizer.word_index
    inv_word_index = {v: k for k, v in word_index.items()}
    print('Done fitting tokenizer on dev texts, found %d unique tokens' % len(word_index))

    print("Tokenizing questions and contexts")
    context_seqs = tokenizer.texts_to_sequences(contexts)
    question_seqs = tokenizer.texts_to_sequences(questions)

    context_seqs_padded = pad_sequences(context_seqs, maxlen=max_context_seq_length, padding='post')
    question_seqs_padded = pad_sequences(question_seqs, maxlen=max_question_seq_length, padding='post')

    assert embedding_dim in [50, 100, 200, 300]
    glove_path = 'glove/glove.6B.%dd.txt' % embedding_dim
    print('Loading glove model')
    glove_model = load_glove_model(glove_path)
    print('Done loading glove model')

    embedding_matrix = get_embedding_matrix(word_index, glove_model)
    model = get_model(embedding_matrix, name='val')

    model.load_weights('simple_bidaf.h5', by_name=True)

    print("Predicting..")
    ps_start, ps_end = model.predict([context_seqs_padded, question_seqs_padded])
    predictions = {}
    for idx, (id, s, e) in enumerate(zip(ids, ps_start, ps_end)):
        i, j = get_best_answer_span(s, e)
        answer = " ".join([inv_word_index[x] for x in context_seqs_padded[idx][i:j] if x])
        predictions[id] = answer

    with open('predictions.txt', 'w') as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
