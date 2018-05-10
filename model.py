from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras.models import Model
import keras


class TextCNN:

    def __init__(self, num_class,
                 num_words,
                 sequence_length,
                 embedding_size,
                 num_filters,
                 filter_sizes):
        # input layer
        sequence_input = Input(shape=(sequence_length,), dtype='int32')
        # embedding layer
        embedding_layer = Embedding(num_words + 1,
                                    embedding_size,
                                    embeddings_initializer=keras.initializers.random_uniform(minval=-0.25, maxval=0.25),
                                    input_length=sequence_length)
        embedded_sequences = embedding_layer(sequence_input)

        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            x = Conv1D(num_filters, filter_size, activation='relu')(embedded_sequences)
            x = MaxPool1D(int(x.shape[1]))(x)
            pooled_outputs.append(x)
        merged = concatenate(pooled_outputs)
        x = Flatten()(merged)
        outputs = Dense(num_class, activation='softmax')(x)
        self.model = Model(sequence_input, outputs)