## defines and trains a feed-forward neural net
## to learn the variance of a Guassian, 
## given the density curve represented as an array of length 200

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import pickle
import argparse 
from tensorflow.random import set_seed
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

# get target shift and seed number
parser = argparse.ArgumentParser()
parser.add_argument('--target_shift', 
        help='amount to element-wise add to targets')
parser.add_argument('--seed', help='tensorflow seed to use')
args = parser.parse_args()

target_shift = float(args.target_shift)
seed = int(args.seed)


# LOAD DATA 
data_file = open('data.pkl', 'rb')
data = pickle.load(data_file)
data_file.close()
# data are the tuple (input, targets) for the neural network
# the input has shape (N, 200) and the targets have shape (N, 1)

# DEFINE MODEL PARAMETERS
in_units = data[0].shape[1]
out_units = data[1].shape[1]
design = {'unit_activations':[(in_units, 'relu'),
                              (70, 'relu'),
                              (40, 'relu'),
                              (10, 'relu'),
                              (out_units, 'relu')],
          'optimizer':'adam',
          'loss':'mse',
          'batch_size':15,
          'epochs':10,
         }

# CONSTRUCT FEED-FORWARD NETWORK
def train_encoder(design, data):

    # unpack design dictionary
    unit_activ = design['unit_activations']
    optim = design['optimizer']
    loss = design['loss']
    batch_size = design['batch_size']
    epochs = design['epochs']

    # build feed forward network
    model = Sequential()
    # construct network
    model.add(Dense(units=unit_activ[0][0],
                    activation=unit_activ[0][1],
                    input_shape=data[0][0].shape))
    model.add(Dense(unit_activ[1][0], unit_activ[1][1]))
    model.add(Dense(unit_activ[2][0], unit_activ[2][1]))
    model.add(Dense(unit_activ[3][0], unit_activ[3][1]))
    model.add(Dense(unit_activ[4][0], unit_activ[4][1]))

    # compile and print summary
    model.compile(optimizer=optim, loss=loss)
    model.summary()

    # shift targets
    shifted_targets = data[1] + target_shift

    # train model
    model.fit(x=data[0], y=shifted_targets,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2)


# TRAIN MODEL
set_seed(seed) 
train_encoder(design, data)
