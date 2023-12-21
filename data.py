import numpy as np
import mnist


x, y, x_test, y_test = mnist.mnist('MNIST')

x_clean = []
y_clean = []

for m in x:
    new_m = m.flatten()
    new_m.shape += (1,)
    x_clean.append(new_m)


for e in y:
    nmb_list = np.zeros(10)
    nmb_list[e] = 1
    nmb_list.shape += (1,)
    y_clean.append(nmb_list)

x_clean = np.array(x_clean)
y_clean = np.array(y_clean)

x_test_clean = []
y_test_clean = []

for m in x_test:
    new_m = m.flatten()
    new_m.shape += (1,)
    x_test_clean.append(new_m)


for e in y_test:
    nmb_list = np.zeros(10)
    nmb_list[e] = 1
    nmb_list.shape += (1,)
    y_test_clean.append(nmb_list)

x_test_clean = np.array(x_test_clean)
y_test_clean = np.array(y_test_clean)

n_train, dim_input, _ = x_clean.shape
_, dim_output, _ = y_clean.shape
n_test,_,_ = x_test.shape