# debug-tensorflow
This is a reproducible bug in `tensorflow.keras` where some additive shifts
in the targets results in the network not learning. 
The data are pickled in `data.pkl` which contains the tuple `(inputs, targets)`
where `inputs` is a numpy array of shape (2000, 200)
and `targets` is a numpy array of shape (2000, 1).

To reproduce the error make sure you have Tensorflow 2.2 installed and run

`python test_encoder.py --target_shift -1.0 --seed 23`

and observe the network does not learn. 
However, if you change the seed to 32, for example, 
while keeping the target shift fixed you will see the network does indeed learn.
Furthermore, if you keep the seed fixed at 23 and change 
the target shift to 0.0 or 1.0 then the network goes back to learning.

The data were simulated by running 

`python make_data.py --seed 23`
