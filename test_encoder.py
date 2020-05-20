## defines and trains a feed-forward neural net
## to learn the variance of a Guassian, 
## given the density curve represented as an array of length 200

import numpy as np 
import pickle
import argparse 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from tensorflow.random import set_seed
import tensorflow.keras as K
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split

# set seed number
parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='tensorflow seed to use')
args = parser.parse_args()
seed = int(args.seed)


# LOAD DATA 
data_file = open('data.pkl', 'rb')
data = pickle.load(data_file)
data_file.close()
# data are the tuple (input, targets) for the neural network
# the input has shape (N, 200) and the targets have shape (N, 1)
x = data[0]
y = data[1]
print('x y shape', x.shape, y.shape)
normalizer = Normalizer()
minmaxscaler = MinMaxScaler()
x = normalizer.fit_transform(x)
y = minmaxscaler.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print('train test x y shape', x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# DEFINE MODEL PARAMETERS
in_units = x_train.shape[1]
out_units = y_train.shape[1]
design = {'unit_activations':[(in_units, 'relu'),
                              (100, 'relu'),
                              (50, 'relu'),
                              (out_units, 'relu')],
          'optimizer':'adam',
          'loss':'mse',
          'batch_size':25,
          'epochs':100,
         }

# CONSTRUCT FEED-FORWARD NETWORK
def train_encoder(design, data):

    # unpack design dictionary
    ua = design['unit_activations']
    optim = design['optimizer']
    loss = design['loss']
    batch_size = design['batch_size']
    epochs = design['epochs']

    # build feed forward network
    model = K.Sequential()
    model.add(K.layers.Dense(units=ua[0][0], activation=ua[0][1],
                             input_shape=data[0][0].shape))
    model.add(K.layers.Dense(ua[1][0], ua[1][1]))
    model.add(K.layers.Dense(ua[2][0], ua[2][1]))
    model.add(K.layers.Dense(ua[3][0], ua[3][1]))

    # compile and print summary
    model.compile(optimizer=optim, loss=loss)
    model.summary()

    # train model
    model.fit(x=data[0], y=data[1],
              batch_size=batch_size,
              epochs=epochs,
              verbose=2)
    return model


# TRAIN MODEL
set_seed(seed) 
model = train_encoder(design, data=(x_train, y_train))
y_pred = model.predict(x_test)
print(y_pred[0:10] - y_test[0:10])
#print('mean predictions', np.mean(target_pred))
#print('predictions variance', np.var(target_pred))
