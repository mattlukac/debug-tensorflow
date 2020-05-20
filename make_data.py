## generates input and targets for feed-forward network
## each input is a Gaussian density with length 200
## each target is the associated Gaussian variance, i.e.
## v in u(x) = 1/sqrt(2pi v) exp(-x^2/(2v))
## The variance is drawn from Uniform(0.1, 1)

import pickle
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='numpy seed to use')
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)

## DOMAIN 
def domain():
    return np.linspace(-1., 1., num=200)

## SOLUTION
def simulate_u(var):
    x = domain()
    return np.exp(-x**2 / (2*var))/np.sqrt(2 * np.pi * var)

## VARIANCE REPLICATES
def simulate_variance(replicates):
    return np.random.uniform(0.1, 1, size=replicates)

## 
def simulate_data(replicates):
    x = domain()
    u_shape = (replicates, ) + x.shape 
    u = np.zeros(u_shape)
    var = simulate_variance(replicates)
    var = var.reshape(replicates, 1)
    for replicate, v in enumerate(var):
        u[replicate] = simulate_u(v)

    return u, var


## check shapes
inputs, targets = simulate_data(2000)
assert inputs.ndim == 2
assert targets.ndim == 2
assert inputs.shape[0] == targets.shape[0]
assert inputs.shape[1] == len(domain())
assert targets.shape[1] == 1

## save data
data = (inputs, targets)
data_file = open('data.pkl', 'wb')
pickle.dump(data, data_file)
data_file.close()

