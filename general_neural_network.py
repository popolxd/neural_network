import numpy as np
from keras.datasets import mnist
import json
import datetime

current_network = 'convolutional.json'
# current_network = 'successful_networks/4-8-pure_cnn - 0.0005.json'
# current_network = 'successful_networks/sigmoid.json'
# current_network = 'general.json'

(train_x, train_y), (test_x, test_y) = mnist.load_data()

with open(current_network, 'r') as f:
    values = json.load(f)

def leaky_ReLU(x):
    new_values = [0] * len(x)

    for i in range(len(x)):
        if x[i] > 0:
            new_values[i] = x[i]
        else:
            new_values[i] = leak_constant * x[i]

    return new_values

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def propagate_forward_ffnn_sigmoid(num_of_layers_without_input, initial_input):
    values_in_between = [initial_input]

    for i in range(num_of_layers_without_input):
        values_in_between.append(
            sigmoid(np.dot(values['weights'][i], values_in_between[i]) + values['biases'][i])
        )

    return values_in_between

def propagate_forward_ffnn_leaky_relu(num_of_layers_without_input, initial_input):
    values_in_between = [initial_input]

    for i in range(num_of_layers_without_input - 1):
        values_in_between.append(
            leaky_ReLU(np.dot(values['weights'][i], values_in_between[i]) + values['biases'][i])
        )
    
    values_in_between.append(
        sigmoid(np.dot(values['weights'][num_of_layers_without_input - 1], values_in_between[num_of_layers_without_input - 1]) + values['biases'][num_of_layers_without_input - 1])
    )

    return values_in_between

def one_layer_backpropagation_sigmoid(na_right_layer, na_middle_layer, na_left_layer, current_layer, gradient_of_right_neurons):
    next_gradient_of_right_neurons = []
    weight_gradients_to_pass_on = []
    bias_gradients_to_pass_on = []

    for i in range(len(na_middle_layer)):
        this_previous_neuron_gradient = 0

        for j in range(len(na_right_layer)):
            this_previous_neuron_gradient += gradient_of_right_neurons[j] * na_right_layer[j] * (1 - na_right_layer[j]) * values['weights'][current_layer][j][i]

        next_gradient_of_right_neurons.append(this_previous_neuron_gradient)

        dc_dz = this_previous_neuron_gradient * na_middle_layer[i] * (1 - na_middle_layer[i])

        weight_gradients_to_pass_on.append([dc_dz * j for j in na_left_layer])
        bias_gradients_to_pass_on.append(dc_dz)

    return {'neuron_gradients': next_gradient_of_right_neurons, 'weight_gradients': weight_gradients_to_pass_on, 'bias_gradients': bias_gradients_to_pass_on}

def one_layer_backpropagation_leaky_relu(na_right_layer, na_middle_layer, na_left_layer, current_layer, gradient_of_right_neurons):
    next_gradient_of_right_neurons = []
    weight_gradients_to_pass_on = []
    bias_gradients_to_pass_on = []

    for i in range(len(na_middle_layer)):
        this_previous_neuron_gradient = 0

        for j in range(len(na_right_layer)):
            if na_right_layer[j] > 0:
                this_previous_neuron_gradient += gradient_of_right_neurons[j] * values['weights'][current_layer][j][i]
            else:
                this_previous_neuron_gradient += gradient_of_right_neurons[j] * leak_constant * values['weights'][current_layer][j][i]

        next_gradient_of_right_neurons.append(this_previous_neuron_gradient)

        if na_middle_layer[i] > 0:
            weight_gradients_to_pass_on.append([this_previous_neuron_gradient * j for j in na_left_layer])
            bias_gradients_to_pass_on.append(this_previous_neuron_gradient)
        else:
            weight_gradients_to_pass_on.append([this_previous_neuron_gradient * leak_constant * j for j in na_left_layer])
            bias_gradients_to_pass_on.append(this_previous_neuron_gradient * leak_constant)

    return {'neuron_gradients': next_gradient_of_right_neurons, 'weight_gradients': weight_gradients_to_pass_on, 'bias_gradients': bias_gradients_to_pass_on}

def last_and_before_that_layer_backpropagation_sigmoid(true_answer, last_neuron_activations, neurons_before_last, neurons_before_before_last):
    desired_output = [1 if i == true_answer else 0 for i in range(10)]

    weight_gradients_to_pass_on_ll = []
    bias_gradients_to_pass_on_ll = []
    weight_gradients_to_pass_on_bl = []
    bias_gradients_to_pass_on_bl = []
    gradient_to_pass_on = []

    for i in range(len(last_neuron_activations)):
        dc_dz = 2 * (last_neuron_activations[i] - desired_output[i]) * last_neuron_activations[i] * (1 - last_neuron_activations[i])

        weight_gradients_to_pass_on_ll.append([dc_dz * j for j in neurons_before_last])
        bias_gradients_to_pass_on_ll.append(dc_dz)

    for i in range(len(neurons_before_last)):
        this_previous_neuron_gradient = 0

        for j in range(len(last_neuron_activations)):
            this_previous_neuron_gradient += 2 * (last_neuron_activations[j] - desired_output[j]) * last_neuron_activations[j] * (1 - last_neuron_activations[j]) * values['weights'][len(values['weights']) - 1][j][i]

        gradient_to_pass_on.append(this_previous_neuron_gradient)

        dc_dz = this_previous_neuron_gradient * neurons_before_last[i] * (1 - neurons_before_last[i])

        weight_gradients_to_pass_on_bl.append([dc_dz * j for j in neurons_before_before_last])
        bias_gradients_to_pass_on_bl.append(dc_dz)

    return {'weight_gradients_ll': weight_gradients_to_pass_on_ll, 
            'bias_gradients_ll': bias_gradients_to_pass_on_ll, 
            'weight_gradients_bl': weight_gradients_to_pass_on_bl, 
            'bias_gradients_bl': bias_gradients_to_pass_on_bl,
            'neuron_gradients': gradient_to_pass_on}

def only_layer_backpropagation(true_answer, last_neuron_activations, neurons_before_last):
    desired_output = [1 if i == true_answer else 0 for i in range(10)]

    weight_gradients_to_pass_on = []
    bias_gradients_to_pass_on = []
    gradient_to_pass_on = []

    for i in range(len(last_neuron_activations)):
        dc_dz = 2 * (last_neuron_activations[i] - desired_output[i]) * last_neuron_activations[i] * (1 - last_neuron_activations[i])

        weight_gradients_to_pass_on.append([dc_dz * j for j in neurons_before_last])
        bias_gradients_to_pass_on.append(dc_dz)

    for i in range(len(neurons_before_last)):
        this_previous_neuron_gradient = 0

        for j in range(len(last_neuron_activations)):
            this_previous_neuron_gradient += 2 * (last_neuron_activations[j] - desired_output[j]) * last_neuron_activations[j] * (1 - last_neuron_activations[j]) * values['weights'][len(values['weights']) - 1][j][i]

        gradient_to_pass_on.append(this_previous_neuron_gradient)

    return {'weight_gradients': weight_gradients_to_pass_on, 
            'bias_gradients': bias_gradients_to_pass_on, 
            'neuron_gradients': gradient_to_pass_on}

def last_and_before_that_layer_backpropagation_leaky_relu(true_answer, last_neuron_activations, neurons_before_last, neurons_before_before_last):
    desired_output = [1 if i == true_answer else 0 for i in range(10)]

    weight_gradients_to_pass_on_ll = []
    bias_gradients_to_pass_on_ll = []
    weight_gradients_to_pass_on_bl = []
    bias_gradients_to_pass_on_bl = []
    gradient_to_pass_on = []

    # last layer stuff

    for i in range(len(last_neuron_activations)):
        dc_dz = 2 * (last_neuron_activations[i] - desired_output[i]) * last_neuron_activations[i] * (1 - last_neuron_activations[i])

        weight_gradients_to_pass_on_ll.append([dc_dz * j for j in neurons_before_last])
        bias_gradients_to_pass_on_ll.append(dc_dz)

    # previous layer stuff

    for i in range(len(neurons_before_last)):
        this_previous_neuron_gradient = 0

        for j in range(len(last_neuron_activations)):
            this_previous_neuron_gradient += 2 * (last_neuron_activations[j] - desired_output[j]) * last_neuron_activations[j] * (1 - last_neuron_activations[j]) * values['weights'][len(values['weights']) - 1][j][i]

        gradient_to_pass_on.append(this_previous_neuron_gradient)

        if neurons_before_last[i] > 0:
            weight_gradients_to_pass_on_bl.append([this_previous_neuron_gradient * j for j in neurons_before_before_last])
            bias_gradients_to_pass_on_bl.append(this_previous_neuron_gradient)
        else:
            weight_gradients_to_pass_on_bl.append([this_previous_neuron_gradient * leak_constant * j for j in neurons_before_before_last])
            bias_gradients_to_pass_on_bl.append(this_previous_neuron_gradient * leak_constant)

    return {'weight_gradients_ll': weight_gradients_to_pass_on_ll, 
            'bias_gradients_ll': bias_gradients_to_pass_on_ll, 
            'weight_gradients_bl': weight_gradients_to_pass_on_bl, 
            'bias_gradients_bl': bias_gradients_to_pass_on_bl,
            'neuron_gradients': gradient_to_pass_on}

def first_layer_backpropagation_sigmoid(na_right_layer, na_middle_layer, na_left_layer, gradient_of_right_neurons):
    weight_gradients_to_pass_on = []
    bias_gradients_to_pass_on = []

    for i in range(len(na_middle_layer)):
        this_previous_neuron_gradient = 0

        for j in range(len(na_right_layer)):
            this_previous_neuron_gradient += gradient_of_right_neurons[j] * na_right_layer[j] * (1 - na_right_layer[j]) * values['weights'][1][j][i]

        dc_dz = this_previous_neuron_gradient * na_middle_layer[i] * (1 - na_middle_layer[i])

        weight_gradients_to_pass_on.append([dc_dz * j for j in na_left_layer])
        bias_gradients_to_pass_on.append(dc_dz)

    return {'weight_gradients': weight_gradients_to_pass_on, 'bias_gradients': bias_gradients_to_pass_on}

def first_layer_backpropagation_leaky_relu(na_right_layer, na_middle_layer, na_left_layer, gradient_of_right_neurons):
    weight_gradients_to_pass_on = []
    bias_gradients_to_pass_on = []

    for i in range(len(na_middle_layer)):
        this_previous_neuron_gradient = 0

        for j in range(len(na_right_layer)):
            if na_right_layer[j] > 0:
                this_previous_neuron_gradient += gradient_of_right_neurons[j] * values['weights'][1][j][i]
            else:
                this_previous_neuron_gradient += gradient_of_right_neurons[j] * leak_constant * values['weights'][1][j][i]

        if na_middle_layer[i] > 0:
            weight_gradients_to_pass_on.append([this_previous_neuron_gradient * j for j in na_left_layer])
            bias_gradients_to_pass_on.append(this_previous_neuron_gradient)
        else:
            weight_gradients_to_pass_on.append([this_previous_neuron_gradient * leak_constant * j for j in na_left_layer])
            bias_gradients_to_pass_on.append(this_previous_neuron_gradient * leak_constant)

    return {'weight_gradients': weight_gradients_to_pass_on, 'bias_gradients': bias_gradients_to_pass_on}

def one_iteration_sigmoid(input, true_answer, num_of_layers_without_input):

    all_neurons_activations = propagate_forward_ffnn_sigmoid(num_of_layers_without_input, input)

    weight_gradients = [0] * num_of_layers_without_input
    bias_gradients = [0] * num_of_layers_without_input

    result = last_and_before_that_layer_backpropagation_sigmoid(true_answer, all_neurons_activations[num_of_layers_without_input], all_neurons_activations[num_of_layers_without_input - 1], all_neurons_activations[num_of_layers_without_input - 2])

    weight_gradients[num_of_layers_without_input - 1] = result['weight_gradients_ll']
    bias_gradients[num_of_layers_without_input - 1] = result['bias_gradients_ll']
    weight_gradients[num_of_layers_without_input - 2] = result['weight_gradients_bl']
    bias_gradients[num_of_layers_without_input - 2] = result['bias_gradients_bl']
    next_neuron_gradients = result['neuron_gradients']

    for i in range(num_of_layers_without_input - 3):
        result = one_layer_backpropagation_sigmoid(all_neurons_activations[num_of_layers_without_input - 1 - i], all_neurons_activations[num_of_layers_without_input - 2 - i], all_neurons_activations[num_of_layers_without_input - 3 - i], num_of_layers_without_input - 2 - i, next_neuron_gradients)
        
        weight_gradients[num_of_layers_without_input - 3 - i] = result['weight_gradients']
        bias_gradients[num_of_layers_without_input - 3 - i] = result['bias_gradients']
        next_neuron_gradients = result['neuron_gradients']

    result = first_layer_backpropagation_sigmoid(all_neurons_activations[2], all_neurons_activations[1], all_neurons_activations[0], next_neuron_gradients)

    weight_gradients[0] = result['weight_gradients']
    bias_gradients[0] = result['bias_gradients']

    return {'weight_gradients': weight_gradients, 'bias_gradients': bias_gradients}

def one_iteration_leaky_relu(input, true_answer, num_of_layers_without_input):
    all_neurons_activations = propagate_forward_ffnn_leaky_relu(num_of_layers_without_input, input)

    weight_gradients = [0] * num_of_layers_without_input
    bias_gradients = [0] * num_of_layers_without_input

    result = last_and_before_that_layer_backpropagation_leaky_relu(true_answer, all_neurons_activations[num_of_layers_without_input], all_neurons_activations[num_of_layers_without_input - 1], all_neurons_activations[num_of_layers_without_input - 2])

    weight_gradients[num_of_layers_without_input - 1] = result['weight_gradients_ll']
    bias_gradients[num_of_layers_without_input - 1] = result['bias_gradients_ll']
    weight_gradients[num_of_layers_without_input - 2] = result['weight_gradients_bl']
    bias_gradients[num_of_layers_without_input - 2] = result['bias_gradients_bl']
    next_neuron_gradients = result['neuron_gradients']

    for i in range(num_of_layers_without_input - 3):
        result = one_layer_backpropagation_leaky_relu(all_neurons_activations[num_of_layers_without_input - 1 - i], all_neurons_activations[num_of_layers_without_input - 2 - i], all_neurons_activations[num_of_layers_without_input - 3 - i], num_of_layers_without_input - 2 - i, next_neuron_gradients)
        
        weight_gradients[num_of_layers_without_input - 3 - i] = result['weight_gradients']
        bias_gradients[num_of_layers_without_input - 3 - i] = result['bias_gradients']
        next_neuron_gradients = result['neuron_gradients']

    result = first_layer_backpropagation_leaky_relu(all_neurons_activations[2], all_neurons_activations[1], all_neurons_activations[0], next_neuron_gradients)

    weight_gradients[0] = result['weight_gradients']
    bias_gradients[0] = result['bias_gradients']

    return {'weight_gradients': weight_gradients, 'bias_gradients': bias_gradients}

def train_sigmoid(batch_size, alpha, alpha_decay, num_of_iters):
    current_alpha = alpha

    start_point = 0
    if num_of_iters < 60000:
        start_point = np.random.randint(0, 60000 - num_of_iters)

    for i in range(num_of_iters):
        new_i = start_point + i

        all_gradients = one_iteration_sigmoid(train_x[new_i].flatten() / 255, train_y[new_i], num_of_layers_without_first)

        if i % batch_size == 0:
            updating_vectors = all_gradients
        else:
            for j in range(len(updating_vectors['bias_gradients'])):
                updating_vectors['bias_gradients'][j] = np.add(updating_vectors['bias_gradients'][j], all_gradients['bias_gradients'][j])
                updating_vectors['weight_gradients'][j] = np.add(updating_vectors['weight_gradients'][j], all_gradients['weight_gradients'][j])

        if i % batch_size == batch_size - 1:

            for j in range(len(updating_vectors['bias_gradients'])):

                values['biases'][j] = np.subtract(values['biases'][j], updating_vectors['bias_gradients'][j] * current_alpha).tolist()
                values['weights'][j] = np.subtract(values['weights'][j], updating_vectors['weight_gradients'][j] * current_alpha).tolist()

            current_alpha *= alpha_decay

def train_leaky_relu(batch_size, alpha, alpha_decay, num_of_iters):
    current_alpha = alpha

    start_point = 0
    if num_of_iters < 60000:
        start_point = np.random.randint(0, 60000 - num_of_iters)

    for i in range(num_of_iters):
        new_i = start_point + i

        all_gradients = one_iteration_leaky_relu(train_x[new_i].flatten() / 255, train_y[new_i], num_of_layers_without_first)

        if i % batch_size == 0:
            updating_vectors = all_gradients
        else:
            for j in range(len(updating_vectors['bias_gradients'])):
                updating_vectors['bias_gradients'][j] = np.add(updating_vectors['bias_gradients'][j], all_gradients['bias_gradients'][j])
                updating_vectors['weight_gradients'][j] = np.add(updating_vectors['weight_gradients'][j], all_gradients['weight_gradients'][j])

        if i % batch_size == batch_size - 1:

            for j in range(len(updating_vectors['bias_gradients'])):

                values['biases'][j] = np.subtract(values['biases'][j], updating_vectors['bias_gradients'][j] * current_alpha).tolist()
                values['weights'][j] = np.subtract(values['weights'][j], updating_vectors['weight_gradients'][j] * current_alpha).tolist()

            current_alpha *= alpha_decay


#################################################################################################################################################

def train_cnn_relu_leaky_relu(batch_size, alpha, alpha_decay, num_of_iters, alpha_cnn):
    current_alpha = alpha
    current_alpha_cnn = alpha_cnn
    printing_num = 0

    start_point = 0
    if num_of_iters < 60000:
        start_point = np.random.randint(0, 60000 - num_of_iters)

    for i in range(num_of_iters):
        new_i = start_point + i

        all_gradients = one_iteration_cnn_relu_leaky_relu(train_x[new_i] / 255, train_y[new_i], num_of_layers_without_first)

        if i % batch_size == 0:
            updating_vectors = all_gradients

            # just printing progress
            if printing_num % (num_of_iters / (batch_size * 10)) == 0:
                print(i / num_of_iters * 100)
            printing_num += 1

        else:
            for j in range(len(updating_vectors['bias_gradients'])):
                updating_vectors['bias_gradients'][j] = np.add(updating_vectors['bias_gradients'][j], all_gradients['bias_gradients'][j])
                updating_vectors['weight_gradients'][j] = np.add(updating_vectors['weight_gradients'][j], all_gradients['weight_gradients'][j])
                
            for j in range(len(updating_vectors['kernel_bias_gradients'])):
                updating_vectors['kernel_bias_gradients'][j] = np.add(updating_vectors['kernel_bias_gradients'][j], all_gradients['kernel_bias_gradients'][j])
                updating_vectors['kernel_gradients'][j] = np.add(updating_vectors['kernel_gradients'][j], all_gradients['kernel_gradients'][j])

        if i % batch_size == batch_size - 1:

            for j in range(len(updating_vectors['bias_gradients'])):
                values['biases'][j] = np.subtract(values['biases'][j], updating_vectors['bias_gradients'][j] * current_alpha).tolist()
                values['weights'][j] = np.subtract(values['weights'][j], updating_vectors['weight_gradients'][j] * current_alpha).tolist()

            for j in range(len(updating_vectors['kernel_bias_gradients'])):
                values['kernel_biases'][j] = np.subtract(values['kernel_biases'][j], updating_vectors['kernel_bias_gradients'][j] * current_alpha_cnn).tolist()
                values['kernels'][j] = np.subtract(values['kernels'][j], updating_vectors['kernel_gradients'][j] * current_alpha_cnn).tolist()

            current_alpha *= alpha_decay
            current_alpha_cnn *= alpha_decay

def relu(x):
    return max(0, x)

# def one_convolution_neuron_output(input, kernels, bias, kernel_size_for_this_layer):
#     return [
#         [
#             relu(np.sum(
#                 [np.sum(
#                     np.multiply(
#                         [
#                             [input[current_input][y + j][x + i] for i in range(kernel_size_for_this_layer)]
#                             for j in range(kernel_size_for_this_layer)
#                         ], kernels[current_input]
#                     )
#                 ) for current_input in range(len(input))]
#             ) + bias)
#             for x in range(len(input[0][0]) - kernel_size_for_this_layer + 1)
#         ] for y in range(len(input[0]) - kernel_size_for_this_layer + 1)
#     ]

def one_convolution_neuron_output(input, kernels, bias, kernel_size_for_this_layer):
    input_array = np.array(input)
    kernels_array = np.array(kernels)

    result = np.zeros((len(input[0]) - kernel_size_for_this_layer + 1,
                       len(input[0][0]) - kernel_size_for_this_layer + 1))

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            result[y, x] = np.sum(np.multiply(
                input_array[:, y:y + kernel_size_for_this_layer, x:x + kernel_size_for_this_layer],
                kernels_array
            )) + bias

    result = np.maximum(0, result)  # ReLU activation

    return result.tolist()

def one_layer_forward_propagation_cnn(input, current_layer_without_input):
    return [
        one_convolution_neuron_output(
            input, values['kernels'][current_layer_without_input][i], values['kernel_biases'][current_layer_without_input][i], kernel_sizes[current_layer_without_input]
        ) for i in range(len(values['kernel_biases'][current_layer_without_input]))
    ]

def pooling_layer(input):
    return [
        [
            [
                max(i[y][x], i[y][x + 1], i[y + 1][x], i[y + 1][x + 1])
                for x in range(0, len(i[y]), 2)
            ] for y in range(0, len(i), 2)
        ] for i in input
    ]

def propagate_forward_cnn_leaky_relu(num_of_cnn_layers, initial_input, num_of_pools):
    values_in_between = [[initial_input]]
    pooled_layers = []

    for i in range(num_of_cnn_layers):
        new_values = one_layer_forward_propagation_cnn(values_in_between[i] if i == 0 or i > num_of_pools else pooled_layers[i - 1], i)
        values_in_between.append(new_values)

        if len(pooled_layers) < num_of_pools:
            pooled_layers.append(pooling_layer(new_values))

    return {'normal_layers': values_in_between, 'pooling_layers': pooled_layers}

def last_layer_from_flat_to_structured(activations):
    num_of_neurons = len(values['kernels'][len(values['kernels']) - 1])
    activation_length = len(activations)
    last_layer_sizes = int(np.sqrt(activation_length // num_of_neurons))
    
    return [
        [
            [
                activations[i*(activation_length // num_of_neurons) + y*last_layer_sizes + x]
                for x in range(last_layer_sizes)
            ] for y in range(last_layer_sizes)
        ] for i in range(num_of_neurons)
    ]

def one_layer_backpropagation_cnn_with_pooling(neuron_activation_gradients, na_middle_layer, na_left_layer_pooled, current_layer):

    all_neuron_biases_gradients_in_this_layer = []
    all_neuron_kernels_gradients_in_this_layer = []
    next_neuron_activation_gradients = [
        [
            [
                0 for x in range(len(na_left_layer_pooled[a][y]))
            ] for y in range(len(na_left_layer_pooled[a]))
        ] for a in range(len(na_left_layer_pooled))
    ]
    
    for i in range(len(neuron_activation_gradients)):

        current_neuron_bias_gradient = 0
        current_neuron_kernels_gradients = [
            [
                [
                    0 for x in range(kernel_sizes[current_layer - 1])
                ] for y in range(kernel_sizes[current_layer - 1])
            ] for j in range(len(values['kernel_biases'][current_layer - 2]))
        ]

        for y in range(len(neuron_activation_gradients[i])):
            for x in range(len(neuron_activation_gradients[i][y])): # podstate prechadzam kazdym Z z outputov jedneho neuronu
                                    # x  y
                before_pooling_pos = []

                max_values_of_this_region = max(na_middle_layer[i][y * 2][x * 2], 
                                                na_middle_layer[i][y * 2][x * 2 + 1],
                                                na_middle_layer[i][y * 2 + 1][x * 2],
                                                na_middle_layer[i][y * 2 + 1][x * 2 + 1])
                
                if max_values_of_this_region == na_middle_layer[i][y * 2][x * 2]:
                    before_pooling_pos = [x * 2, y * 2]
                elif max_values_of_this_region == na_middle_layer[i][y * 2][x * 2 + 1]:
                    before_pooling_pos = [x * 2 + 1, y * 2]
                elif max_values_of_this_region == na_middle_layer[i][y * 2 + 1][x * 2]:
                    before_pooling_pos = [x * 2, y * 2 + 1]
                else:
                    before_pooling_pos = [x * 2 + 1, y * 2 + 1]

                # here i have correct position of the actual neuron before pooling
                
                if na_middle_layer[i][before_pooling_pos[1]][before_pooling_pos[0]] > 0:
                    current_neuron_bias_gradient += neuron_activation_gradients[i][y][x]

                    # kernel gradients and next gradients
                    for a in range(len(current_neuron_kernels_gradients)):

                        for k_y in range(kernel_sizes[current_layer - 1]):
                            for k_x in range(kernel_sizes[current_layer - 1]):

                                current_neuron_kernels_gradients[a][k_y][k_x] += neuron_activation_gradients[i][y][x] * na_left_layer_pooled[a][before_pooling_pos[1] + k_y][before_pooling_pos[0] + k_x]
                                next_neuron_activation_gradients[a][before_pooling_pos[1] + k_y][before_pooling_pos[0] + k_x] += neuron_activation_gradients[i][y][x] * values['kernels'][current_layer - 1][i][a][k_y][k_x]

        all_neuron_biases_gradients_in_this_layer.append(current_neuron_bias_gradient)
        all_neuron_kernels_gradients_in_this_layer.append(current_neuron_kernels_gradients)

    return {'neuron_gradients': next_neuron_activation_gradients, 'weight_gradients': all_neuron_kernels_gradients_in_this_layer, 'bias_gradients': all_neuron_biases_gradients_in_this_layer}

def first_layer_backpropagation_cnn_with_pooling(neuron_activation_gradients, na_middle_layer, na_left_layer_pooled, current_layer):
    all_neuron_biases_gradients_in_this_layer = []
    all_neuron_kernels_gradients_in_this_layer = []
    
    for i in range(len(neuron_activation_gradients)):

        current_neuron_bias_gradient = 0
        current_neuron_kernels_gradients = [
            [
                [
                    0 for x in range(kernel_sizes[current_layer - 1])
                ] for y in range(kernel_sizes[current_layer - 1])
            ] for j in range(1)
        ]

        for y in range(len(neuron_activation_gradients[i])):
            for x in range(len(neuron_activation_gradients[i][y])): # podstate prechadzam kazdym Z z outputov jedneho neuronu
                                    # x  y
                before_pooling_pos = []

                max_values_of_this_region = max(na_middle_layer[i][y * 2][x * 2], 
                                                na_middle_layer[i][y * 2][x * 2 + 1],
                                                na_middle_layer[i][y * 2 + 1][x * 2],
                                                na_middle_layer[i][y * 2 + 1][x * 2 + 1])
                
                if max_values_of_this_region == na_middle_layer[i][y * 2][x * 2]:
                    before_pooling_pos = [x * 2, y * 2]
                elif max_values_of_this_region == na_middle_layer[i][y * 2][x * 2 + 1]:
                    before_pooling_pos = [x * 2 + 1, y * 2]
                elif max_values_of_this_region == na_middle_layer[i][y * 2 + 1][x * 2]:
                    before_pooling_pos = [x * 2, y * 2 + 1]
                else:
                    before_pooling_pos = [x * 2 + 1, y * 2 + 1]

                # here i have correct position of the actual neuron before pooling
                
                if na_middle_layer[i][before_pooling_pos[1]][before_pooling_pos[0]] > 0:
                    current_neuron_bias_gradient += neuron_activation_gradients[i][y][x]

                    # kernel gradients and next gradients
                    for a in range(len(current_neuron_kernels_gradients)):

                        for k_y in range(kernel_sizes[current_layer - 1]):
                            for k_x in range(kernel_sizes[current_layer - 1]):

                                current_neuron_kernels_gradients[a][k_y][k_x] += neuron_activation_gradients[i][y][x] * na_left_layer_pooled[a][before_pooling_pos[1] + k_y][before_pooling_pos[0] + k_x]

        all_neuron_biases_gradients_in_this_layer.append(current_neuron_bias_gradient)
        all_neuron_kernels_gradients_in_this_layer.append(current_neuron_kernels_gradients)

    return {'weight_gradients': all_neuron_kernels_gradients_in_this_layer, 'bias_gradients': all_neuron_biases_gradients_in_this_layer}

def last_layer_backpropagation_cnn_with_pooling(neuron_activation_gradients, na_middle_layer, na_left_layer_pooled, current_layer):
    return one_layer_backpropagation_cnn_with_pooling(last_layer_from_flat_to_structured(neuron_activation_gradients), na_middle_layer, na_left_layer_pooled, current_layer)

def one_iteration_cnn_relu_leaky_relu(input, true_answer, num_of_layers_without_input, num_of_cnn_layers=2, num_of_pools=2):

    all_cnn_activations = propagate_forward_cnn_leaky_relu(num_of_cnn_layers, input, num_of_pools)
    all_ffnn_activations = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, np.array(all_cnn_activations['pooling_layers'][len(all_cnn_activations) - 1]).flatten())

    weight_gradients = [0] * num_of_layers_without_input
    bias_gradients = [0] * num_of_layers_without_input
    kernel_gradients = [0] * num_of_cnn_layers
    kernel_biases_gradients = [0] * num_of_cnn_layers

    if num_of_layers_without_input > 1:
        result = last_and_before_that_layer_backpropagation_leaky_relu(true_answer, all_ffnn_activations[num_of_layers_without_input], all_ffnn_activations[num_of_layers_without_input - 1], all_ffnn_activations[num_of_layers_without_input - 2])

        weight_gradients[num_of_layers_without_input - 1] = result['weight_gradients_ll']
        bias_gradients[num_of_layers_without_input - 1] = result['bias_gradients_ll']
        weight_gradients[num_of_layers_without_input - 2] = result['weight_gradients_bl']
        bias_gradients[num_of_layers_without_input - 2] = result['bias_gradients_bl']
        next_neuron_gradients = result['neuron_gradients']

        for i in range(num_of_layers_without_input - 2):
            result = one_layer_backpropagation_leaky_relu(all_ffnn_activations[num_of_layers_without_input - 1 - i], all_ffnn_activations[num_of_layers_without_input - 2 - i], all_ffnn_activations[num_of_layers_without_input - 3 - i], num_of_layers_without_input - 2 - i, next_neuron_gradients)
            
            weight_gradients[num_of_layers_without_input - 3 - i] = result['weight_gradients']
            bias_gradients[num_of_layers_without_input - 3 - i] = result['bias_gradients']
            next_neuron_gradients = result['neuron_gradients']
    
    else: # just one layer, i could also add option to classic feed forward backpropagation, but who tf uses one layer.

        result = only_layer_backpropagation(true_answer, all_ffnn_activations[1], all_ffnn_activations[0])

        weight_gradients[0] = result['weight_gradients']
        bias_gradients[0] = result['bias_gradients']
        next_neuron_gradients = result['neuron_gradients']
    
    # cnn backpropagation
    
    result = last_layer_backpropagation_cnn_with_pooling(
        next_neuron_gradients, all_cnn_activations['normal_layers'][len(all_cnn_activations['normal_layers']) - 1], all_cnn_activations['pooling_layers'][len(all_cnn_activations['pooling_layers']) - 2], num_of_cnn_layers
    )

    kernel_gradients[num_of_cnn_layers - 1] = result['weight_gradients']
    kernel_biases_gradients[num_of_cnn_layers - 1] = result['bias_gradients']
    next_neuron_gradients = result['neuron_gradients']

    for i in range(num_of_cnn_layers - 2):
        result = one_layer_backpropagation_cnn_with_pooling(
            next_neuron_gradients, all_cnn_activations['normal_layers'][len(all_cnn_activations['normal_layers']) - 2 - i], all_cnn_activations['pooling_layers'][len(all_cnn_activations['pooling_layers']) - 3 - i], num_of_cnn_layers - 1 - i
        )

        kernel_gradients[num_of_cnn_layers - 2 - i] = result['weight_gradients']
        kernel_biases_gradients[num_of_cnn_layers - 2 - i] = result['bias_gradients']
        next_neuron_gradients = result['neuron_gradients']

    result = first_layer_backpropagation_cnn_with_pooling(
        next_neuron_gradients, all_cnn_activations['normal_layers'][1], [input], 1
    )

    kernel_gradients[0] = result['weight_gradients']
    kernel_biases_gradients[0] = result['bias_gradients']

    return {
        'weight_gradients': weight_gradients,
        'bias_gradients': bias_gradients,
        'kernel_gradients': kernel_gradients,
        'kernel_bias_gradients': kernel_biases_gradients
    }

#################################################################################################################################################

def calculate_one_cost_and_accuracy(output, true_answer):
    cost = 0
    highest_value = 0
    current_answer = -1

    for i in range(len(output)):
        if i != true_answer:
            cost += (output[i] ** 2)
        else:
            cost += ((1 - output[i]) ** 2)

        if output[i] > highest_value:
            highest_value = output[i]
            current_answer = i
    
    return {'cost': cost, 'is_right': 1 if current_answer == true_answer else 0}

def test(current_function, test_size):
    total_error = 0
    total_accuracy = 0

    offset = 0
    if test_size < 10000:
        offset = np.random.randint(0, 10000 - test_size)

    if current_function == 'sigmoid':
        for i in range(test_size):

            result = propagate_forward_ffnn_sigmoid(num_of_layers_without_first, test_x[offset + i].flatten() / 255)
            data = calculate_one_cost_and_accuracy(result[len(result) - 1], test_y[offset + i])

            total_error += data['cost']
            total_accuracy += data['is_right']
    
    elif current_function == 'leaky relu':
        for i in range(test_size):

            result = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, test_x[offset + i].flatten() / 255)
            data = calculate_one_cost_and_accuracy(result[len(result) - 1], test_y[offset + i])

            total_error += data['cost']
            total_accuracy += data['is_right']

    elif current_function == 'cnn relu leaky relu':
        for i in range(test_size):

            cnn_result = propagate_forward_cnn_leaky_relu(num_of_cnn_layers, test_x[offset + i] / 255, num_of_cnn_layers)
            result = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, np.array(cnn_result['pooling_layers'][len(cnn_result) - 1]).flatten())

            data = calculate_one_cost_and_accuracy(result[len(result) - 1], test_y[offset + i])

            total_error += data['cost']
            total_accuracy += data['is_right']

    return {'error': total_error / test_size, 'accuracy': total_accuracy / test_size * 100}

def debug_test_full_size(current_function, test_size):
    total_error = 0
    total_accuracy = 0

    offset = 0
    if test_size < 10000:
        offset = np.random.randint(0, 60000 - test_size)

    if current_function == 'sigmoid':
        for i in range(test_size):

            result = propagate_forward_ffnn_sigmoid(num_of_layers_without_first, train_x[offset + i].flatten() / 255)
            data = calculate_one_cost_and_accuracy(result[len(result) - 1], train_y[offset + i])

            total_error += data['cost']
            total_accuracy += data['is_right']

    elif current_function == 'leaky relu':
        for i in range(test_size):

            result = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, train_x[offset + i].flatten() / 255)
            data = calculate_one_cost_and_accuracy(result[len(result) - 1], train_y[offset + i])

            total_error += data['cost']
            total_accuracy += data['is_right']

    elif current_function == 'cnn relu leaky relu':
        for i in range(test_size):

            cnn_result = propagate_forward_cnn_leaky_relu(num_of_cnn_layers, train_x[offset + i] / 255, num_of_cnn_layers)
            result = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, np.array(cnn_result['pooling_layers'][len(cnn_result) - 1]).flatten())

            data = calculate_one_cost_and_accuracy(result[len(result) - 1], train_y[offset + i])

            total_error += data['cost']
            total_accuracy += data['is_right']

    return {'error': total_error / test_size, 'accuracy': total_accuracy / test_size * 100}

def train_full_size(typeof_function, batch_size, alpha, alpha_decay, alpha_cnn=0):
    time_before = datetime.datetime.now()

    if typeof_function == 'sigmoid':
        train_sigmoid(batch_size, alpha, alpha_decay, 60000)
    elif typeof_function == 'leaky relu':
        train_leaky_relu(batch_size, alpha, alpha_decay, 60000)
    elif typeof_function == 'cnn relu leaky relu':
        train_cnn_relu_leaky_relu(batch_size, alpha, alpha_decay, 60000, alpha_cnn)

    time_after = datetime.datetime.now()
    print(time_after - time_before)

def train_partially(typeof_function, batch_size, alpha, alpha_decay, num_of_iters, alpha_cnn=0):
    time_before = datetime.datetime.now()

    if typeof_function == 'sigmoid':
        train_sigmoid(batch_size, alpha, alpha_decay, num_of_iters)
    elif typeof_function == 'leaky relu':
        train_leaky_relu(batch_size, alpha, alpha_decay, num_of_iters)
    elif typeof_function == 'cnn relu leaky relu':
        train_cnn_relu_leaky_relu(batch_size, alpha, alpha_decay, num_of_iters, alpha_cnn)

    time_after = datetime.datetime.now()
    print(time_after - time_before)

# allways relevant
num_of_layers_without_first = len(values['weights'])
current_function = 'cnn relu leaky relu'
# current_function = 'leaky relu'

batch_size = 10
alpha = 0.003
alpha_cnn = alpha * 1

alpha_decay = 1

# relevant with leaky relu 
leak_constant = 0.01

# relevant with cnn
if current_function == 'cnn relu leaky relu':
    kernel_sizes = [len(values['kernels'][i][0][0]) for i in range(len(values['kernels']))]
    num_of_cnn_layers = len(values['kernel_biases'])
    num_of_pools = num_of_cnn_layers * 1
    num_of_neurons_in_each_cnn_layer = [len(values['kernel_biases'][i]) for i in range(len(values['kernel_biases']))]

if_train = input('1 - train full size, 2 - test, 3 - initialize, 4 - test base set, 5 - train partially, 6 - visualize: ')

if if_train == '1':

    train_full_size(current_function, batch_size, alpha, alpha_decay, alpha_cnn)

    with open(current_network, 'w') as f:
        json.dump(values, f, indent=2)

elif if_train == '2':
    test_size = int(input('test size: '))
    print(test(current_function, test_size))

elif if_train == '3':

    values = {
        'weights': [],
        'biases': []
    }

    activation_function = input('1 - sigmoid, 2 - leaky relu:, 3 - cnn relu ')
    num_of_layers_without_first_input = int(input('number of layers without input one: '))
    layers = [784]

    for i in range(num_of_layers_without_first_input):
        layers.append(int(input('number of neurons in ' + str(i + 1) + '. layer: ')))

    if activation_function == '1':

        for i in range(len(layers)):

            if i != len(layers) - 1:
                values['weights'].append([])

                for j in range(layers[i + 1]):
                    values['weights'][i].append([])

                    for k in range(layers[i]):
                        values['weights'][i][j].append(np.random.randn() * np.sqrt(2 / (layers[i] + layers[i + 1])))

            if i != 0:
                values['biases'].append([])

                for j in range(layers[i]):
                    values['biases'][i - 1].append(0)

    elif activation_function == '2':

        for i in range(len(layers)):

            if i != len(layers) - 1:
                values['weights'].append([])

                for j in range(layers[i + 1]):
                    values['weights'][i].append([])

                    for k in range(layers[i]):
                        values['weights'][i][j].append(np.random.randn() * np.sqrt(2 / layers[i]))

            if i != 0:
                values['biases'].append([])

                for j in range(layers[i]):
                    values['biases'][i - 1].append(0.1)

    elif activation_function == '3':
        values = {
            'kernels': [],
            'kernel_biases': [],
            'weights': [],
            'biases': [],
        }

        num_of_cnn_layers = int(input('number of convolutional layers: (for now input two otherwise it will break) '))
        num_of_pools = num_of_cnn_layers * 1
        num_of_neurons_in_each_cnn_layer = []
        kernel_sizes = []

        for i in range(num_of_cnn_layers):
            num_of_neurons_in_each_cnn_layer.append(int(input('number of neurons in ' + str(i + 1) + '. layer: ')))
            kernel_sizes.append(int(input('kernel size in ' + str(i + 1) + '. layer: ')))

        layers[0] = 25 * num_of_neurons_in_each_cnn_layer[len(num_of_neurons_in_each_cnn_layer) - 1]

        for i in range(len(layers)):

            if i != len(layers) - 1:
                values['weights'].append([])

                for j in range(layers[i + 1]):
                    values['weights'][i].append([])

                    for k in range(layers[i]):
                        values['weights'][i][j].append(np.random.randn() * np.sqrt(2 / layers[i]))

            if i != 0:
                values['biases'].append([])

                for j in range(layers[i]):
                    values['biases'][i - 1].append(0.1)

        for i in range(num_of_cnn_layers):
            values['kernel_biases'].append([])
            values['kernels'].append([])
            
            for j in range(num_of_neurons_in_each_cnn_layer[i]):
                values['kernel_biases'][i].append(0)
                values['kernels'][i].append([])

                for k in range(1 if i == 0 else num_of_neurons_in_each_cnn_layer[i - 1]):
                    values['kernels'][i][j].append([]) # now i am at one kernel

                    for y in range(kernel_sizes[i]):
                        values['kernels'][i][j][k].append([])

                        for x in range(kernel_sizes[i]):
                            values['kernels'][i][j][k][y].append(np.random.randn() * np.sqrt(2 / 784) if i == 0 else np.random.randn() * np.sqrt(2 / (144 * num_of_neurons_in_each_cnn_layer[i - 1]))) # test, from num of inputs

    with open(current_network, 'w') as f:
        json.dump(values, f, indent=2)

elif if_train == '4':
    test_size = int(input('test size: '))
    print(debug_test_full_size(current_function, test_size))

elif if_train == '5':
    num_of_iters = int(input('num of iterations: '))
    train_partially(current_function, batch_size, alpha, alpha_decay, num_of_iters, alpha_cnn)

    with open(current_network, 'w') as f:
        json.dump(values, f, indent=2)

elif if_train == '6':
    import pygame
    pygame.init()

    width = 1280
    height = 720

    input_size = 28 * 20
    square_size = 20

    current_input_image = np.array([[0 for i in range(28)] for j in range(28)])
    current_right_answer = -1

    win = pygame.display.set_mode((width, height))
    run = True
    clock = pygame.time.Clock()
    fps = 60
    hold = False

    font = pygame.font.SysFont('Calibri', 40)
    small_font = pygame.font.SysFont('Calibri', 25)
    font_color = (230, 230, 230)

    generate_random_image_rect = pygame.Rect(20, 40 + input_size, 100, 40)
    compute_answer_rect = pygame.Rect(input_size - 80, 40 + input_size, 100, 40)
    check_answer_rect = pygame.Rect(input_size - 80, 100 + input_size, 100, 40)
    painting_thickness_rect = pygame.Rect(input_size + 180, 20, 40, 40)

    paint_rect = pygame.Rect(input_size + 40, 20, 40, 40)
    erase_rect = pygame.Rect(input_size + 40, 80, 40, 40)

    painting = False
    painting_thickness = 1.5

    erasing = False
    erasing_thickness = 3

    current_cnn_layer_displayed = 0
    current_set_of_neurons_displayed = 0
    displaying_weights = True

    color_const_for_kernels = 1
    color_const_for_outputs = 0.5
    color_const_for_first_layer_weights = 5

    change_set_of_neurons_displayed_rect = pygame.Rect(width - 160, 100, 60, 60)
    current_set_of_neurons_text = small_font.render(str(current_set_of_neurons_displayed*4 + 1) + ' - ' + str(current_set_of_neurons_displayed*4 + 4), False, font_color)

    change_layer_displayed_rect = pygame.Rect(width - 80, 100, 60, 60)
    current_cnn_layer_displayed_text = font.render(str(current_cnn_layer_displayed + 1), False, font_color)

    change_displayed_thing_rect = pygame.Rect(width - 240, 100, 60, 60)

    all_current_outputs = None

    nn_answer_text = font.render('', False, font_color)
    my_answer = None
    correct_answer_text = font.render('', False, font_color)
    painting_thickness_text = font.render(str(painting_thickness), False, font_color)

    def draw_window():
        win.fill((0, 0, 0))

        for y in range(len(current_input_image)):
            for x in range(len(current_input_image[y])):
                pygame.draw.rect(win, (max(20, current_input_image[y][x]), max(20, current_input_image[y][x]), max(20, current_input_image[y][x])), (x * square_size + 20, y * square_size + 20, square_size - 1, square_size - 1))

        win.blit(nn_answer_text, (280, 40 + input_size))
        win.blit(correct_answer_text, (280, 100 + input_size))
        win.blit(painting_thickness_text, (input_size + 180, 80))

        pygame.draw.rect(win, (230, 20, 20), generate_random_image_rect)
        pygame.draw.rect(win, (20, 230, 20), compute_answer_rect)
        pygame.draw.rect(win, (20, 20, 230), check_answer_rect)
        pygame.draw.rect(win, (20, 230, 20) if painting else (70, 70, 70), paint_rect)
        pygame.draw.rect(win, (20, 230, 20) if erasing else (70, 70, 70), erase_rect)

        pygame.draw.rect(win, (30, 30, 30), (input_size + 100, 30, 200, 10))
        pygame.draw.rect(win, (128, 128, 128), painting_thickness_rect)

        pygame.draw.rect(win, (50, 50, 50), change_set_of_neurons_displayed_rect)
        win.blit(current_set_of_neurons_text, (width - 150, 60))

        if current_function == 'cnn relu leaky relu':

            pygame.draw.rect(win, (50, 50, 50), change_layer_displayed_rect)
            win.blit(current_cnn_layer_displayed_text, (width - 60, 110))

            pygame.draw.rect(win, (50, 50, 50), change_displayed_thing_rect)

            if displaying_weights:
                for neuron in range(current_set_of_neurons_displayed*4, min(len(values['kernels'][current_cnn_layer_displayed]), current_set_of_neurons_displayed*4 + 4)):
                    for kernel in range(len(values['kernels'][current_cnn_layer_displayed][neuron])):

                        for y in range(len(values['kernels'][current_cnn_layer_displayed][neuron][kernel])):
                            for x in range(len(values['kernels'][current_cnn_layer_displayed][neuron][kernel][y])):

                                current_value = values['kernels'][current_cnn_layer_displayed][neuron][kernel][y][x]

                                pygame.draw.rect(
                                    win, 
                                    (np.tanh(current_value * color_const_for_kernels) * 255, 0, 0) if current_value >= 0 else (0, 0, np.tanh(-current_value * color_const_for_kernels) * 255), 
                                    (input_size + 100 + (neuron % 4)*150 + x*20, 200 + kernel*100 + y*20, 19, 19)
                                )

            else: # displaying outputs
                for neuron in range(current_set_of_neurons_displayed*4, min(len(all_current_outputs['pooling_layers'][current_cnn_layer_displayed]), current_set_of_neurons_displayed*4 + 4)):

                    for y in range(len(all_current_outputs['pooling_layers'][current_cnn_layer_displayed][neuron])):
                        for x in range(len(all_current_outputs['pooling_layers'][current_cnn_layer_displayed][neuron][y])):

                            current_value = all_current_outputs['pooling_layers'][current_cnn_layer_displayed][neuron][y][x]
                            one_color_value = np.tanh(current_value * color_const_for_outputs) * 255

                            pygame.draw.rect(
                                win,
                                (one_color_value, one_color_value, one_color_value),
                                (input_size + 100 + (neuron % 4)*150 + x * 10, 200 + y * 10, 9, 9)
                            )
        
        else: # here only thing that really can be visualized are weights from input to fist layer
            for neuron in range(current_set_of_neurons_displayed*4, min(len(values['weights'][0]), current_set_of_neurons_displayed*4 + 4)):

                for y in range(28):
                    for x in range(28):

                        current_value = values['weights'][0][neuron][y*28 + x]

                        pygame.draw.rect(
                            win,
                            (np.tanh(current_value * color_const_for_first_layer_weights) * 255, 0, 0) if current_value >= 0 else (0, 0, np.tanh(-current_value * color_const_for_first_layer_weights) * 255),
                            (input_size + 100 + (neuron % 4)*150 + x * 5, 200 + y * 5, 4, 4)
                        )

        pygame.display.update()

    def move(click, mouse, hold):
        global current_input_image, current_right_answer, nn_answer_text, correct_answer_text, my_answer, painting, erasing, painting_thickness, painting_thickness_text, current_set_of_neurons_text, current_set_of_neurons_displayed, current_cnn_layer_displayed, current_cnn_layer_displayed_text, displaying_weights, all_current_outputs

        if hold:
            if pygame.Rect.collidepoint(pygame.Rect(20, 20, input_size - 1, input_size - 1), mouse):
                rounded_painting_thickness = int(painting_thickness) + 1

                if painting:
                    square_touched = ((mouse[0] - 20) // square_size, (mouse[1] - 20) // square_size)

                    for y in range(max(0, square_touched[1] - rounded_painting_thickness), min(28, square_touched[1] + rounded_painting_thickness + 1)):
                        for x in range(max(0, square_touched[0] - rounded_painting_thickness), min(28, square_touched[0] + rounded_painting_thickness + 1)):

                            distance = np.sqrt((mouse[0] - 20 - (x*square_size + square_size / 2)) ** 2 + (mouse[1] - 20 - (y*square_size + square_size / 2)) ** 2)

                            if distance <= painting_thickness * square_size:
                                color_value = (painting_thickness * square_size - distance) / (painting_thickness * square_size) * 255

                                current_input_image[y][x] = max(color_value, current_input_image[y][x])

                elif erasing:
                    square_touched = ((mouse[0] - 20) // square_size, (mouse[1] - 20) // square_size)

                    for y in range(max(0, square_touched[1] - erasing_thickness), min(28, square_touched[1] + erasing_thickness + 1)):
                        for x in range(max(0, square_touched[0] - erasing_thickness), min(28, square_touched[0] + erasing_thickness + 1)):

                            current_input_image[y][x] = 0

            elif pygame.Rect.collidepoint(pygame.Rect(input_size + 120, 20, 160, 40), mouse):
                    painting_thickness_rect.x = mouse[0] - 20

                    painting_thickness = (painting_thickness_rect.x - input_size - 100) / 160 * 2 + 0.5
                    painting_thickness_text = font.render(str(round(painting_thickness, 2)), False, font_color)

        if click:
            if pygame.Rect.collidepoint(generate_random_image_rect, mouse):
                random_num = np.random.randint(0, 10000)
                current_input_image = test_x[random_num]
                current_right_answer = test_y[random_num]

            elif pygame.Rect.collidepoint(compute_answer_rect, mouse):
                result = -1

                if current_function == 'sigmoid':
                    result = propagate_forward_ffnn_sigmoid(num_of_layers_without_first, current_input_image.flatten() / 255)
                elif current_function == 'leaky relu':
                    result = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, current_input_image.flatten() / 255)
                elif current_function == 'cnn relu leaky relu':
                    cnn_result = propagate_forward_cnn_leaky_relu(num_of_cnn_layers, current_input_image / 255, num_of_cnn_layers)
                    result = propagate_forward_ffnn_leaky_relu(num_of_layers_without_first, np.array(cnn_result['pooling_layers'][len(cnn_result['pooling_layers']) - 1]).flatten())
                    all_current_outputs = cnn_result

                result_as_num = -1
                highest_value = -1

                for i in range(10):
                    if result[len(result) - 1][i] > highest_value:
                        result_as_num = i
                        highest_value = result[len(result) - 1][i]

                nn_answer_text = font.render(str(result_as_num), False, font_color)
                my_answer = result_as_num

            elif pygame.Rect.collidepoint(check_answer_rect, mouse):
                correct_answer_text = font.render(str(current_right_answer), False, font_color)

            elif pygame.Rect.collidepoint(paint_rect, mouse):
                painting = not painting
                erasing = False

            elif pygame.Rect.collidepoint(erase_rect, mouse):
                painting = False
                erasing = not erasing

            elif pygame.Rect.collidepoint(change_set_of_neurons_displayed_rect, mouse):
                if current_function == 'cnn relu leaky relu':
                    if len(values['kernel_biases'][current_cnn_layer_displayed]) > current_set_of_neurons_displayed*4 + 4:
                        current_set_of_neurons_displayed += 1
                    else:
                        current_set_of_neurons_displayed = 0
                
                else:
                    if len(values['weights'][0]) > current_set_of_neurons_displayed*4 + 4:
                        current_set_of_neurons_displayed += 1
                    else:
                        current_set_of_neurons_displayed = 0

                current_set_of_neurons_text = small_font.render(str(current_set_of_neurons_displayed*4 + 1) + ' - ' + str(current_set_of_neurons_displayed*4 +4), False, font_color)

            elif pygame.Rect.collidepoint(change_layer_displayed_rect, mouse) and current_function == 'cnn relu leaky relu':
                if len(values['kernel_biases']) > current_cnn_layer_displayed + 1:
                    current_cnn_layer_displayed += 1
                else:
                    current_cnn_layer_displayed = 0
                current_set_of_neurons_displayed = 0

                current_set_of_neurons_text = small_font.render(str(current_set_of_neurons_displayed*4 + 1) + ' - ' + str(current_set_of_neurons_displayed*4 +4), False, font_color)
                current_cnn_layer_displayed_text = font.render(str(current_cnn_layer_displayed + 1), False, font_color)

            elif pygame.Rect.collidepoint(change_displayed_thing_rect, mouse) and current_function == 'cnn relu leaky relu':
                if (displaying_weights and all_current_outputs) != None or not displaying_weights:
                    displaying_weights = not displaying_weights

    while run:
        clock.tick(fps)
        click = False
        mouse = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                click = True
                hold = True
            if event.type == pygame.MOUSEBUTTONUP:
                hold = False

        draw_window()
        move(click, mouse, hold)

    pygame.quit()