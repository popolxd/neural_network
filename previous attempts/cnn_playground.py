import numpy as np
from keras.datasets import mnist

values = {
    'kernel_biases': [
        [0, 0],
        [0, 0, 0, 0]
    ],
    'kernels': [
        [
            [
                [
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1]
                ]
            ],
            [
                [
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-1, -1, -1, -1, -1]
                ]
            ]
        ],
        [
            [
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ],
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ]
            ],
            [
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ],
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ]
            ],
            [
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ],
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ]
            ],
            [
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ],
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]
                ]
            ]
        ]
    ]
}

(train_x, train_y), (test_x, test_y) = mnist.load_data()
kernel_sizes = [5, 3]
num_of_pools = 2

def relu(x):
    return max(0, x)

def one_convolution_neuron_output(input, kernels, bias, kernel_size_for_this_layer):
    return [
        [
            relu(np.sum(
                [np.sum(
                    np.multiply(
                        [
                            [input[current_input][y + j][x + i] for i in range(kernel_size_for_this_layer)]
                            for j in range(kernel_size_for_this_layer)
                        ], kernels[current_input]
                    )
                ) for current_input in range(len(input))]
            ) + bias)
            for x in range(len(input[0][0]) - kernel_size_for_this_layer + 1)
        ] for y in range(len(input[0]) - kernel_size_for_this_layer + 1)
    ]
            

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

        all_neuron_biases_gradients_in_this_layer.append(current_neuron_bias_gradient)
        all_neuron_kernels_gradients_in_this_layer.append(current_neuron_kernels_gradients)

    return {'weight_gradients': all_neuron_kernels_gradients_in_this_layer, 'bias_gradients': all_neuron_biases_gradients_in_this_layer}

def last_layer_backpropagation_cnn_with_pooling(neuron_activation_gradients, na_middle_layer, na_left_layer, current_layer):
    return one_layer_backpropagation_cnn_with_pooling(last_layer_from_flat_to_structured(neuron_activation_gradients), na_middle_layer, na_left_layer, current_layer)

def one_iteration_cnn(num_of_cnn_layers, initial_input):
    test_gradients = [np.random.random() - 0.5 for i in range(100)]
    all_cnn_values = propagate_forward_cnn_leaky_relu(num_of_cnn_layers, initial_input, 2)

    cnn_weight_gradients = []
    cnn_bias_gradients = []

    # THIS WORKS ONLY WITH ALL POOLING LAYERS!
    result = last_layer_backpropagation_cnn_with_pooling(test_gradients, all_cnn_values['normal_layers'][len(all_cnn_values['normal_layers']) - 1], all_cnn_values['pooling_layers'][len(all_cnn_values['pooling_layers']) - 2], 2)

##################################################################################################################################################
##################################################################################################################################################

def one_convolution_neuron_output(input, kernels, bias, kernel_size_for_this_layer):
    return [
        [
            relu(np.sum(
                [np.sum(
                    np.multiply(
                        [
                            [input[current_input][y + j][x + i] for i in range(kernel_size_for_this_layer)]
                            for j in range(kernel_size_for_this_layer)
                        ], kernels[current_input]
                    )
                ) for current_input in range(len(input))]
            ) + bias)
            for x in range(len(input[0][0]) - kernel_size_for_this_layer + 1)
        ] for y in range(len(input[0]) - kernel_size_for_this_layer + 1)
    ]

def one_convolution_neuron_output(input, kernels, bias, kernel_size_for_this_layer):
    input_array = np.array(input)
    kernels_array = np.array(kernels)

    result = np.zeros((len(input[0]) - kernel_size_for_this_layer + 1,
                       len(input[0][0]) - kernel_size_for_this_layer + 1))

    for i in range(len(kernels)):
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                result[y, x] += np.sum(np.multiply(
                    input_array[:, y:y + kernel_size_for_this_layer, x:x + kernel_size_for_this_layer],
                    kernels_array[i]
                ))

    result += bias
    result = np.maximum(0, result)  # ReLU activation

    return result.tolist()