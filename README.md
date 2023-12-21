Hello ! This is my attempt to implement a CNN (convolutional neural network) using an evolution strategy, the Covariance Matrix Adapation algorithm (cma).  
  
Feel free to use my code :)  
  
HOW CMA WORKS : imagine you can only see f(x_i) for any x_i, you can't access to f, you can't have its gradient, its a noisy function, etc... (f is called a black box)  
But, you want to find its argmin anyway. The ecma evol. strat. is able to do so, by creating a new generation of (x_i)_i<l according to a gaussian law, and select the ones,  
lets call them (x_i*)_i<m with m<l, such that f(x_i*)_i are the smallest one. Then, you update the mean, the standard deviation and the covariance matrix for the next  
generation, until, you hit a stopping criterium.  
  
HOW TO DEAL WITH NN : rather than nudging the parameters (weights and biases) according to the gradient descent of the cost function with backward prop, here I create a new generation of   
flattened parameters that follows the cma algorithm to evolve towards one of the cost function's local minimum (the global one if ur lucky lol). 
  
MY CODE CONTAINS:  
 1) data : I use here the mnist handwritten digits db, it is really NOT adapted to cma because the number of inputs is quite large. CMA is quasi-newtonian, but as the dimension increases,  
 the complexity explodes (works well with dim <~200)  
 2) ploting : to provide a graphic example
 3) nn_struct : because cma works quite differently, I had to add original features on my CNN to make it easier to code. There is lots of things to explain so hmu if you want details,
    also contains a bonus example CNN from scratch that uses backprop
 5) evolv=_strat_cma : contains the algorithm that takes as input the function to minimize and some hyperparameters, it returns the argmin of f
    CAREFUL : there is still some arbitrary truncations for the condition in the while loop : its because the algorithm is taking WAY TOO MUCH time (g<1000, you should remove that)
 6) neuroevolution : that's where I use cma on the CNN, note that the cost function is defined relatively to the nature of the algorithm (takes x as an array but x is the weights
    and biases flattened, which isn't easy to use for forward prop)

FOR NOW : it works ! ~10h of calcuations, for 1000 generations, for an average prediction success rate of ~50% on the mnist's test db. One could think it's not good, but remember mnist is 
really not adapted for cma, so getting 50% on only 1000 generations shows it converges pretty well.
FOR NEXT : will push an autoencoder to reduce the input's dimension (28x28 pixels = 784, and remember the input for the cma isn't the input but the weights and biases, so if my first hidden 
layer contains 10 neurons, the dim of input for cma is at leats a matrix of 784x10 weights)
