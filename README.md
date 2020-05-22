# When keeping it ReLU goes wrong
Here I argue that using ReLU activations is not wise when
the goal is to train a neural network to approximate a real-valued
function. This is demonstrated in the `relu_pitfalls.ipynb` file.

Consider a feed-forward network with a single hidden layer,
containing n hidden units. If there is a single input node
and the bias vectors are initialized to zero, there are cases where
the initialized weight matrices (in this case they are column and row vectors)
will result in no activation in the final layer, thus preventing the 
network from learning. In this setting the main case to consider is 
whether or not the angle between the weight vectors is acute or not.

Assuming all inputs are positive and the two activation functions 
are the identity map f(x) = x, the final node will be proportional
to the dot product of the weight vectors. So if we have ReLU activations
and all hidden units are activated, but the weight vectors have an
obtuse angle between them in n-dimensional space, then the final unit
will never be activated, thus impeding the learning process of the neural net.
