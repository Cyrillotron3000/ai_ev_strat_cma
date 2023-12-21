import numpy as np
import data

#For this evolv. strat. problem I needed a structure for my neural network where I can do :
#   1) Easy access to weights and biases
#   2) Transfrom my network's parameters (list of matrices and list of biases) 
#      into a 1-D vector, and go back and forth between the forms in a bijective way
#   3) Keep in memory as much variables as I can (layers when forward prop, etc...)

#Neural Network building ------------------------------------------------------

class NeuralNetwork:
    def __init__(self, n_input, n_output, hidden_layers):
        self.n_intput = n_input
        self.n_output = n_output

        hid_layz = []
        m = len(hidden_layers)
        for i in range(0, m):
            hid_layz.append(np.zeros(hidden_layers[i]))
        for e in hid_layz:
            e.shape += (1,)

        self.hidden_layers = hid_layz

        w = [np.random.uniform(-0.5,0.5,(hidden_layers[0], n_input))]
        b = [np.random.uniform(-0.5,0.5,(hidden_layers[0]))]
        for i in range(0, m-1):
            w.append(np.random.uniform(-0.5,0.5,(hidden_layers[i+1], hidden_layers[i])))
            b.append(np.random.uniform(-0.5,0.5,(hidden_layers[i+1])))
        w.append(np.random.uniform(-0.5,0.5,(n_output, hidden_layers[m-1])))
        b.append(np.random.uniform(-0.5,0.5,(n_output)))
        for e in b:
            e.shape += (1,)
        
        self.weights = w
        self.biases = b
        self.length = m
        self.output_layer = np.zeros((n_output,1))
        self.weights_shape = [w.shape for w in self.weights]
        self.biases_shape = [b.shape for b in self.biases]
        self.array_form = np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        self.array_form.shape += (1,)


    def infos(self):
        print("dim in: ", self.n_intput)
        print("dim out: ", self.n_output)
        print("hidden layers: ", [l.shape for l in self.hidden_layers])
        print("weights: ", [mat.shape for mat in self.weights])
        print("biases: ", [b.shape for b in self.biases])

    def forward_prop(self, input, f_act, f_out):
        self.hidden_layers[0] = f_act((self.weights[0] @ input) + self.biases[0])
        for i in range(1,self.length):
            self.hidden_layers[i] = f_act((self.weights[i] @ self.hidden_layers[i-1]) + self.biases[i])
        self.output_layer = f_out((self.weights[self.length] @ self.hidden_layers[self.length-1]) + self.biases[self.length])

    def to_array(self):
        self.array_form = np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        self.array_form.shape += (1,)

    def to_w_and_b(self):
        total_params = sum(np.prod(shape) for shape in self.weights_shape) + sum(np.prod(shape) for shape in self.biases_shape)
        if len(self.array_form) != total_params:
            raise ValueError("Incorrect number of parameters")
        weight_params = self.array_form[:sum(np.prod(shape) for shape in self.weights_shape)]
        bias_params = self.array_form[sum(np.prod(shape) for shape in self.weights_shape):]
        self.weights = [w.reshape(shape) for w, shape in zip(np.split(weight_params, np.cumsum([np.prod(shape) for shape in self.weights_shape])[:-1]), self.weights_shape)]
        self.biases = [b.reshape(shape) for b, shape in zip(np.split(bias_params, np.cumsum([np.prod(shape) for shape in self.biases_shape])[:-1]), self.biases_shape)]

        
        
# END Neural Network building ------------------------------------------------------

      
# Example : classic gradient descent with mnist db
        
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -300, 300)))

def soft_max(x):
    s = np.sum(np.exp(np.clip(x, -300, 300)))
    return np.exp(np.clip(x, -300, 300))/s

def back_prop_mse(nn, input, target, lr, f_act, f_out):
    nn.forward_prop(input, f_act, f_out)
    m = nn.length

    delta_out = nn.output_layer - target
    nn.weights[m] += -lr * delta_out @ np.transpose(nn.hidden_layers[m-1])
    nn.biases[m] += -lr * delta_out

    prev_delta_l = delta_out
    for i in range(m-1,0,-1):
        delta_l = np.transpose(nn.weights[i+1]) @ prev_delta_l * (nn.hidden_layers[i] * (1 - nn.hidden_layers[i]))
        nn.weights[i] += -lr * delta_l @ np.transpose(nn.hidden_layers[i-1])
        nn.biases[i] += -lr * delta_l
        prev_delta_l = delta_l
    delta_in = np.transpose(nn.weights[1]) @ prev_delta_l * (nn.hidden_layers[0] * (1 - nn.hidden_layers[0]))
    nn.weights[0] += -lr * delta_in @ np.transpose(input)
    nn.biases[0] += -lr * delta_in

def gradient_descent_mse(iter, lr, f_act, f_out, epochs):
    nn = NeuralNetwork(784,10,[10])
    for k in range(epochs):
        print("epoch: ",k)
        for i in range(0, iter):
            back_prop_mse(nn, data.x_clean[i], data.y_clean[i], lr, f_act, f_out)
    return nn


#test with lr = 0.01 , activation function = function for ouputs = sigmoid, over 1000 epochs
#careful it can take more than an hour to calculate depending on ur machine

nn_test = gradient_descent_mse(data.n_train -1, 0.01, sigmoid, sigmoid, 15)

counter = 0
for i in range(0,data.n_test-1):
    nn_test.forward_prop(data.x_test_clean[i], sigmoid, sigmoid)
    if np.argmax(nn_test.output_layer) == np.argmax(data.y_test_clean[i]):
        counter +=1
print(100*counter/(data.n_test-1))

#END Example ------------------------------------------------------