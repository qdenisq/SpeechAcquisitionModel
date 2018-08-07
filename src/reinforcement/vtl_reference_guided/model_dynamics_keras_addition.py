from random import seed
from random import randint
from random import random as rnd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np

# generate examples of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest, n_dim):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [[(rnd() - 0.5)*largest for n in range(n_dim)] for _ in range(n_numbers)]
        out_pattern = np.subtract(in_pattern[1], in_pattern[0])
        X.append(in_pattern)
        y.append(out_pattern)
    # format as NumPy arrays
    X, y = array(X), array(y)
    # normalize
    X = X.astype('float') / float(largest * n_numbers)
    y = y.astype('float') / float(largest * n_numbers)
    return X, y


# invert normalization
def invert(value, n_numbers, largest):
    return np.rint(value * float(largest * n_numbers))


# generate training data
seed(1)
n_examples = 100
n_numbers = 2
largest = 1000
n_dim = 20
# define LSTM configuration
n_batch = 2
n_epoch = 50
# create LSTM
model = Sequential()
model.add(Dense(4 * n_dim, input_dim=n_numbers * n_dim))
model.add(Dense(2 * n_dim))
model.add(Dense(1 * n_dim))
model.compile(loss='mean_squared_error', optimizer='adam')
# train LSTM
for _ in range(n_epoch):
    X, y = random_sum_pairs(n_examples, n_numbers, largest, n_dim)
    X_flat = np.reshape(X, [n_examples, n_dim * n_numbers])
    model.fit(X_flat, y, epochs=1, batch_size=n_batch, verbose=2)
# evaluate on some new patterns
X, y = random_sum_pairs(n_examples, n_numbers, largest, n_dim)
X_flat = np.reshape(X, [n_examples, n_dim * n_numbers])
result = model.predict(X_flat, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
    error = expected[i] - predicted[i]
    print('Expected={}, Predicted={} (err={})'.format(expected[i], predicted[i], error))