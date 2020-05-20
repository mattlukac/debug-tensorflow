# Stochastic Failure of a Feed-Forward Network
Here is some unexpected behavior in `tensorflow.keras`.
The data are pickled in `data.pkl` which contains the tuple `(inputs, targets)`
where `inputs` is a numpy array of shape (2000, 200)
and `targets` is a numpy array of shape (2000, 1).
Here, the `targets` are variances for the `inputs`, which represent
normal distributions (with mean 0) on the interval [-1,1].
If we normalize the targets so they are contained in the unit interval,
and set the tensorflow seed to 23, the network will not learn anything.

To reproduce the behavior make sure you have Tensorflow 2.2 installed and run

`python test_encoder.py --seed 23`

and observe the network does not learn. 
If instead you use the flag `--seed 666` the network goes back to learning.

The data were simulated by running 

`python make_data.py --seed 23`
