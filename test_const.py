import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# simulate and split data
# here the samples are constant functions f(x) = c
n_samples = 2000 
n_features = 200
x = np.ones((n_samples, n_features))
np.random.seed(23)
y = np.random.uniform(0.1, 1.0, size=n_samples)
for idx in range(n_samples):
    x[idx] *= y[idx]
print('check constants:')
print('  x1', x[0])
print('  y1', y[0])
print('  x2', x[1])
print('  y2', y[1])

x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=0.1, 
        random_state=23)

# build model
tf.random.set_seed(23)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(
    units=16, 
    activation='relu', 
    input_shape=(n_features,)))
model.add(tf.keras.layers.Dense(12, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=15, batch_size=25)
