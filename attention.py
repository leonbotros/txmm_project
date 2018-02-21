from keras import backend as K
from keras.activations import softmax
from keras.engine.topology import Layer


class Attention(Layer):
    """
    Simpler version of the Attention layer proposed by Seo et al.
    The call() function calculates G matrix in the paper using dot product,
    so this layer has no trainable weights.
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        tensor1_dim = input_shape[0][-1]
        tensor2_dim = input_shape[1][-1]
        assert tensor1_dim == tensor2_dim
        self.trainable_weights = []

    def call(self, inputs, **kwargs):
        encoded_context, encoded_question = inputs # [batch_size, N, 2l], [batch_size, M, 2l]
        num_rows_1 = K.shape(encoded_context)[1]
        num_rows_2 = K.shape(encoded_question)[1]
        tile_dims_1 = K.concatenate([[1, 1], [num_rows_2], [1]], 0) # [1, 1, M, 1]
        tile_dims_2 = K.concatenate([[1], [num_rows_1], [1, 1]], 0) # [1, N, 1, 1]
        encoded_context_expanded = K.expand_dims(encoded_context, axis=2) # [batch_size, N, 1, 2l]
        encoded_question_expanded = K.expand_dims(encoded_question, axis=1) # [batch_size, 1, M, 2l]
        tiled_matrix_1 = K.tile(encoded_context_expanded, tile_dims_1) # [batch_size, N, M, 2l]
        tiled_matrix_2 = K.tile(encoded_question_expanded, tile_dims_2) # [batch_size, N, M, 2l]

        # vectorwise dot product as similarity
        similarity_matrix = K.sum(tiled_matrix_1 * tiled_matrix_2, axis=-1) # [batch_size, N, M]

        # apply softmax over columns of similarity matrix
        context_query_attention_vector = softmax(similarity_matrix, axis=-1) # [batch_size, N,]

        # Calculate weighted question vector
        num_attention_dims = K.ndim(context_query_attention_vector)
        num_matrix_dims = K.ndim(encoded_question) - 1
        for _ in range(num_attention_dims - num_matrix_dims):
            encoded_question = K.expand_dims(encoded_question, axis=1)
        weighted_question_vector = K.sum(K.expand_dims(context_query_attention_vector, axis=-1) * encoded_question, -2) # [batch_size, N, 2L]

        # concatenate the encoded question with the weighted question vector
        G = K.concatenate([encoded_context, weighted_question_vector], axis=-1) # [batch_Size, N, 4L]
        return G

    def compute_output_shape(self, input_shape):
        h_input_shape = input_shape[0] # [batch_size, N, 2L]
        u_input_shape = input_shape[1] # [batch_size, M, 2L]
        if h_input_shape[-1] == u_input_shape[-1]:
            return (input_shape[0][0], h_input_shape[1], 2 * h_input_shape[-1]) # [batch_size, N, 4L]
