import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

def load_data_and_labels(file_dir):
    files = os.listdir(file_dir)
    label_index = {}
    labels = []
    data = []
    max_sequence_length = 0
    for file in files:
        label_id = len(label_index)
        label_index[file] = label_id
        for line in open(os.path.join(file_dir, file),'r', encoding='utf-8'):
            if len(line.split()) > max_sequence_length:
                max_sequence_length = len(line.split())
            data.append(line)
            labels.append(label_id)
    # print(data[0])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    data = pad_sequences(sequences, max_sequence_length, padding='post')
    return np.array(data), to_categorical(labels), len(tokenizer.word_index)


if __name__ == '__main__':
    x, y, num_words = load_data_and_labels('./data')
    print(x[0])
    print(np.array(x).shape, np.array(y).shape)
