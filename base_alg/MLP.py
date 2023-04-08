import random

import numpy as np


def create_data(data_dim: int, data_set_size: int):
    ds = {
        'x': [],
        'y': []
    }
    for i in range(data_set_size):
        x = np.array([random.randint(-10, 10) for j in range(data_dim)])
        y = sum(x)  # we want to learn sum operation
        ds['x'].append(x)
        ds['y'].append(y)
    return ds


def initialize_parameters(net_struct: dict):
    input_size = net_struct['input_layer_size']
    '''lets assume we know we have two hidden layers'''
    hidden_1_size = net_struct['hidden_layers_size'][0]
    hidden_2_size = net_struct['hidden_layers_size'][1]
    output_size = net_struct['output_layer_size']
    ''''''
    w_inp_to_l1 = np.random.random(size=[input_size, hidden_1_size])
    w_l1_l2 = np.random.random(size=[hidden_1_size, hidden_2_size])
    w_l2_out = np.random.random(size=[hidden_2_size, output_size])

    b_l1 = np.zeros([hidden_1_size])
    b_l2 = np.zeros([hidden_2_size])

    parameters = {
        "w_inp_to_l1": w_inp_to_l1,
        "w_l1_l2": w_l1_l2,
        "w_l2_out": w_l2_out,
        "b_l1": b_l1,
        "b_l2": b_l2
    }
    return parameters


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_der(z):
    z * (1.0 - z)


def relu(z):
    return np.array([max(x, 0) for x in z])


def feed_forward(x, parameters: dict):
    w_inp_to_l1 = parameters['w_inp_to_l1']
    w_l1_l2 = parameters['w_l1_l2']
    w_l2_out = parameters['w_l2_out']
    b_l1 = parameters['b_l1']
    b_l2 = parameters['b_l2']

    '''calculation'''
    z_1 = np.dot(x, w_inp_to_l1) + b_l1
    a_1 = sigmoid(z_1)

    z_2 = np.dot(a_1, w_l1_l2) + b_l2
    a_2 = sigmoid(z_2)

    z_out = np.dot(a_2, w_l2_out)
    # out = sigmoid(z_out) # For classification
    return z_out[0]


if __name__ == '__main__':
    data_dim = 2
    ds_tr = create_data(data_dim=data_dim, data_set_size=100)
    ds_ts = create_data(data_dim=data_dim, data_set_size=10)
    '''create network'''
    net_struct = {
        'input_layer_size': data_dim,
        'hidden_layers_size': [3, 2],
        'output_layer_size': 1
    }

    ''' create model'''
    parameters = initialize_parameters(net_struct)
    '''train'''
    for i in range(1000):
        x = ds_tr['x'][i]
        y = ds_tr['y'][i]
        y_hat = feed_forward(x, parameters)
        '''loss'''
        loss = np.square(y - y_hat)
        pass
