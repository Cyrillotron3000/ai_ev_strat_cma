import numpy as np
import time
import data
import nn_struct
import evol_strat_cma

#The cost function remains the classic mse, and we still use the mnist database
#Its not really adapted as n=dim(input) is quite large (~8000? all the weigths and biases), 
# the aim is to compare accuracy when the gen counter is arbitrarily stopped

def mse(x,y):
    s = 0
    for ex,ey in zip(x,y):
        s += (ex - ey)**2
    return s

#easier to call the sigmoid this way

def sgmd(x):
    return nn_struct.sigmoid(x)

def es_model():
    #init nn
    nn = nn_struct.NeuralNetwork(784,10,[10])

    #define cost relatively to the nn
    def f(x):
        nn.array_form = x
        avg = 0
        nn.to_w_and_b()
        for i in range(0,int(np.floor(data.n_train*0.2))):
            nn.forward_prop(data.x_clean[i], sgmd,sgmd)
            y_hat = nn.output_layer
            avg += (1/data.n_train)*mse(y_hat,data.y_clean[i])
        return avg
    
    #init params
    x_0 = nn.array_form
    sigma_0 = 1
    pop_sup = 0
    fit_crit = 10**(-3)

    #this might take a while lol
    nn.array_form = evol_strat_cma.cma_es(x_0,sigma_0,pop_sup,f,fit_crit)

    nn.to_w_and_b()

    return nn

s = time.time()

nn_test = es_model()

e = time.time()

count = 0
for (x,y) in zip(data.x_test_clean,data.y_test):
    nn_test.forward_prop(x, sgmd, sgmd)
    if np.argmax(nn_test.output_layer) == y:
        count += 1

print("accuracy: ", 100*count/data.n_test, "% & time for 1000 gen: ", e-s, " s")