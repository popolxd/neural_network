import numpy as np
from keras.datasets import mnist
import json

current_network = "xavier_init.json"
if_train = input('Do you wanna train the network: (random key: Yes, n: No) ')

with open(current_network, 'r') as f:
    values = json.load(f)

(train_x, train_y), (test_x, test_y) = mnist.load_data()

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def multiply_matrixes(current_layer, previous_activations):

    if current_layer < 3:

        new_activations = sigmoid(np.dot(values['weights'][current_layer], previous_activations) + values['biases'][current_layer])

        if current_layer == 2:
            return [previous_activations, multiply_matrixes(current_layer + 1, new_activations)]
        else:
            return [previous_activations, *multiply_matrixes(current_layer + 1, new_activations)]
    
    else:
        return previous_activations

def one_datapiece_backpropagation(neurons_activations, true_answer):
    weight_gradients = [[], [], []]
    bias_gradients = [[], [], []]

    current_activations_gradient = []
    desired_output = []

    for i in range(10):
        if i == true_answer:
            desired_output.append(1)
        else:
            desired_output.append(0)

    ### LAST LAYER OPERATIONS ###

    for i in range(len(neurons_activations[3])):
        weight_gradients[2].append([])

        for j in range(len(neurons_activations[2])):
            weight_gradients[2][i].append(
                2 * (neurons_activations[3][i] - desired_output[i]) * neurons_activations[3][i] * (1 - neurons_activations[3][i]) * neurons_activations[2][j]
            )

        bias_gradients[2].append(
            2 * (neurons_activations[3][i] - desired_output[i]) * neurons_activations[3][i] * (1 - neurons_activations[3][i])
        )

    for i in range(len(neurons_activations[2])):
        this_neuron_gradient = 0

        for j in range(len(neurons_activations[3])):
            this_neuron_gradient += 2 * (neurons_activations[3][j] - desired_output[j]) * neurons_activations[3][j] * (1 - neurons_activations[3][j]) * values['weights'][2][j][i]

        current_activations_gradient.append(this_neuron_gradient)

    ### SECOND HIDDEN LAYER OPERATIONS ###

    for i in range(len(neurons_activations[2])):
        weight_gradients[1].append([])

        for j in range(len(neurons_activations[1])):
            weight_gradients[1][i].append(
                current_activations_gradient[i] * neurons_activations[2][i] * (1 - neurons_activations[2][i]) * neurons_activations[1][j]
            )
        
        bias_gradients[1].append(
            current_activations_gradient[i] * neurons_activations[2][i] * (1 - neurons_activations[2][i])
        )

    for i in range(len(neurons_activations[1])):
        this_neuron_gradient = 0

        for j in range(len(neurons_activations[2])):
            this_neuron_gradient += current_activations_gradient[j] * neurons_activations[2][j] * (1 - neurons_activations[2][j]) * values['weights'][1][j][i]

        if i + 1 <= len(current_activations_gradient):
            current_activations_gradient[i] = this_neuron_gradient
        else:
            current_activations_gradient.append(this_neuron_gradient)

    ### FIRST HIDDEN LAYER OPERATIONS

    for i in range(len(neurons_activations[1])):
        weight_gradients[0].append([])

        for j in range(len(neurons_activations[0])):
            weight_gradients[0][i].append(
                current_activations_gradient[i] * neurons_activations[1][i] * (1 - neurons_activations[1][i]) * neurons_activations[0][j]
            )

        bias_gradients[0].append(
            current_activations_gradient[i] * neurons_activations[1][i] * (1 - neurons_activations[1][i])
        )

    return {
        'weight_gradients': weight_gradients,
        'bias_gradients': bias_gradients
    }

def calculate_one_cost(output, true_answer):
    cost = 0

    for i in range(len(output)):
        if i != true_answer:
            cost += (output[i] ** 2)
        else:
            cost += ((1 - output[i]) ** 2)
    
    return cost

def is_right(output, true_answer):
    highest_value = 0
    current_answer = -1

    for i in range(len(output)):
        if output[i] > highest_value:
            highest_value = output[i]
            current_answer = i

    if current_answer == true_answer:
        return 1
    else:
        return 0
    
def train_network(alpha, alpha_decay, batch_size, num_of_steps):
    updating_vectors = None
    current_alpha = alpha

    rand_num = np.random.randint(0, 60000 / batch_size)

    for i in range(rand_num * batch_size, (rand_num + num_of_steps) * batch_size):

        result = multiply_matrixes(0, train_x[i % 60000].flatten() / 255)

        current_vectors = one_datapiece_backpropagation(result, train_y[i % 60000])

        if (i - rand_num * batch_size) % ((num_of_steps * batch_size) / 10) == 0:
            print(str((i - rand_num * batch_size) / (num_of_steps * batch_size) * 100) + '%')

        if i % batch_size == 0:
            updating_vectors = current_vectors
        else:
            for j in range(len(updating_vectors['bias_gradients'])):
                updating_vectors['bias_gradients'][j] = np.add(updating_vectors['bias_gradients'][j], current_vectors['bias_gradients'][j])
                updating_vectors['weight_gradients'][j] = np.add(updating_vectors['weight_gradients'][j], current_vectors['weight_gradients'][j])

        if i % batch_size == batch_size - 1:

            for j in range(len(updating_vectors['bias_gradients'])):
                values['biases'][j] = np.subtract(values['biases'][j], updating_vectors['bias_gradients'][j] * current_alpha).tolist()
                values['weights'][j] = np.subtract(values['weights'][j], updating_vectors['weight_gradients'][j] * current_alpha).tolist()

        current_alpha *= alpha_decay

    ## SAVING THE LEARNING PROGRESS

    with open(current_network, 'w') as f:
        json.dump(values, f, indent=2)

def calculate_error(num_of_exaples):
    error = 0
    accuracy = 0

    if num_of_exaples < 10000:
        rand_num = np.random.randint(0, 10000 - num_of_exaples)
    else:
        rand_num = 0

    for i in range(rand_num, rand_num + num_of_exaples):
        result = multiply_matrixes(0, test_x[i].flatten() / 255)

        error += calculate_one_cost(result[len(result) - 1], test_y[i])
        accuracy += is_right(result[len(result) - 1], test_y[i])

    return [error / num_of_exaples, accuracy / num_of_exaples * 100]

if if_train != 'n':
    alpha = 0.025
    alpha_decay = 1
    batch_size = 50
    num_of_steps = 1200

    train_network(alpha, alpha_decay, batch_size, num_of_steps)

else:
    initialize_weights = input('What to do? (initialize weights: i, test error: t, big_test: b, play around: random input) ')
    
    if initialize_weights == 'i':

        is_xavier = input('wanna use xavier initialization? (y: Yes. random input: No)')
        first_hidden_layer = int(input('number of neurons in 1. hidden layer: '))
        second_hidden_layer = int(input('number of neurons in 2. hidden layer: '))

        non = [784, first_hidden_layer, second_hidden_layer, 10]

        values = {
            'weights': [],
            'biases': []
        }

        if is_xavier == 'y':
            for i in range(4):

                if i != 3:
                    values['weights'].append([])

                    for j in range(non[i + 1]):
                        values['weights'][i].append([])

                        for k in range(non[i]):
                            values['weights'][i][j].append(np.random.randn() * np.sqrt(1 / non[i]))

                if i != 0:
                    values['biases'].append([])

                    for j in range(non[i]):
                        values['biases'][i - 1].append(0)

        else:
            for i in range(4):

                if i != 3:
                    values['weights'].append([])

                    for j in range(non[i + 1]):
                        values['weights'][i].append([])

                        for k in range(non[i]):
                            values['weights'][i][j].append(np.random.randn())

                if i != 0:
                    values['biases'].append([])

                    for j in range(non[i]):
                        values['biases'][i - 1].append(0)
        
        with open(current_network, 'w') as f:
            json.dump(values, f, indent=2)

    elif initialize_weights == 't' or initialize_weights == 'b':
        num = 500
        if initialize_weights == 'b':
            num = 10000

        result = calculate_error(num)

        print('average error: ' + str(result[0]))
        print('accuracy: ' + str(result[1]) + '%')

    else:
        import pygame

        pygame.init()
        pygame.font.init()

        win = pygame.display.set_mode((420, 520))

        def draw_window(click, mouse):
            global image, text_surface

            win.fill((0, 0, 0))

            for y in range(28):
                for x in range(28):
                    pygame.draw.rect(win, (max(20, image[y][x]), max(20, image[y][x]), max(20, image[y][x])), (x * 15, y * 15, 14, 14))

            generate_image_in_set = pygame.draw.rect(win, (30, 30, 220), (0, 420, 100, 100))
            generate_answer = pygame.draw.rect(win, (220, 30, 30), (320, 420, 100, 100))
            win.blit(text_surface, (195, 440))
            
            if click:
                if pygame.Rect.colliderect(generate_image_in_set, mouse):
                    rand_num = np.random.randint(0, 10000)
                    image = test_x[rand_num]

                elif pygame.Rect.colliderect(generate_answer, mouse):
                    result = multiply_matrixes(0, np.array(image).flatten() / 255)
                    highest_result = 0
                    answer = None

                    for i in range(len(result[len(result) - 1])):
                        if result[len(result) - 1][i] > highest_result:
                            highest_result = result[len(result) - 1][i]
                            answer = i

                    text_surface = my_font.render(str(answer), False, (230, 230, 230))

            pygame.display.update()

        run = True
        clock = pygame.time.Clock()
        click = False

        my_font = pygame.font.SysFont('calibry', 50)
        text_surface = my_font.render('', False, (230, 230, 230))

        image = []

        for i in range(28):
            image.append([])
            for j in range(28):
                image[i].append(0)

        while run:

            clock.tick(60)
            click = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    click = True

            pos = pygame.mouse.get_pos()
            mouse = pygame.Rect(pos[0], pos[1], 1, 1)
            draw_window(click, mouse)

        pygame.quit()