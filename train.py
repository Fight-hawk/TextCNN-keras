from __future__ import print_function
import numpy as np
from data_helper import load_data_and_labels
from model import TextCNN


VALIDATION_SPLIT = 0.1
CORPUS_DIR = './data'
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_SIZE = 256
NUM_FILTERS = 128
FILTER_SIZES = [3, 4, 5]

# Load data and labels
data, labels, num_words = load_data_and_labels(CORPUS_DIR)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Training model.')

text_cnn = TextCNN(num_class=y_train.shape[1],
                   num_words=num_words,
                   sequence_length=data.shape[1],
                   embedding_size=EMBEDDING_SIZE,
                   num_filters=NUM_FILTERS,
                   filter_sizes=FILTER_SIZES)
model = text_cnn.model
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# evaluate
score = model.evaluate(x_train, y_train, verbose=0)
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_val,  y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])