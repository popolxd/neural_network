import numpy as np
from keras.datasets import mnist
import json
import datetime

current_network = "testing.json"
if_train = input('Do you wanna train the network: (random key: Yes, n: No) ')
leak_constant = 0.01

with open(current_network, 'r') as f:
    values = json.load(f)

(train_x, train_y), (test_x, test_y) = mnist.load_data()

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def relu(x):
    new_activations = []

    for i in x:
        if i > 0:
            new_activations.append(i)
        else:
            new_activations.append(leak_constant * i)

    return new_activations

def multiply_matrixes(current_layer, previous_activations):

    if current_layer < 3:

        if current_layer == 2:
            new_activations = sigmoid(np.dot(values['weights'][current_layer], previous_activations) + values['biases'][current_layer])
            return [previous_activations, multiply_matrixes(current_layer + 1, new_activations)]
        else:
            new_activations = relu(np.dot(values['weights'][current_layer], previous_activations) + values['biases'][current_layer])
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

        dc_dz = 2 * (neurons_activations[3][i] - desired_output[i]) * neurons_activations[3][i] * (1 - neurons_activations[3][i])

        weight_gradients[2].append([dc_dz * neurons_activations[2][j] for j in range(len(neurons_activations[2]))])

        bias_gradients[2].append(dc_dz)

    ### SECOND HIDDEN LAYER OPERATIONS ### tuto uz neni sigmoid

    for i in range(len(neurons_activations[2])):

        this_neuron_gradient = 0

        for j in range(len(neurons_activations[3])):
            this_neuron_gradient += 2 * (neurons_activations[3][j] - desired_output[j]) * neurons_activations[3][j] * (1 - neurons_activations[3][j]) * values['weights'][2][j][i]

        current_activations_gradient.append(this_neuron_gradient)

        if neurons_activations[2][i] > 0:
            weight_gradients[1].append([this_neuron_gradient * neurons_activations[1][j] for j in range(len(neurons_activations[1]))])

            bias_gradients[1].append(
                this_neuron_gradient
            )

        else:
            weight_gradients[1].append([this_neuron_gradient * leak_constant * neurons_activations[1][j] for j in range(len(neurons_activations[1]))])
            
            bias_gradients[1].append(
                this_neuron_gradient * leak_constant
            )

    ### FIRST HIDDEN LAYER OPERATIONS

    for i in range(len(neurons_activations[1])):
        this_neuron_gradient = 0

        for j in range(len(neurons_activations[2])):
            if neurons_activations[2][j] > 0:
                this_neuron_gradient += current_activations_gradient[j] * values['weights'][1][j][i]
            else:
                this_neuron_gradient += current_activations_gradient[j] * leak_constant * values['weights'][1][j][i]

        if neurons_activations[1][i] > 0:
            weight_gradients[0].append([this_neuron_gradient * neurons_activations[0][j] for j in range(len(neurons_activations[0]))])

            bias_gradients[0].append(
                this_neuron_gradient
            )
        
        else:
            weight_gradients[0].append([this_neuron_gradient * leak_constant * neurons_activations[0][j] for j in range(len(neurons_activations[0]))])

            bias_gradients[0].append(
                this_neuron_gradient * leak_constant
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

    starting_loc = rand_num * batch_size
    num_of_iter = num_of_steps * batch_size

    for i in range(starting_loc, starting_loc + num_of_iter):

        result = multiply_matrixes(0, train_x[i % 60000].flatten() / 255)

        current_vectors = one_datapiece_backpropagation(result, train_y[i % 60000])

        if (i - starting_loc) % (num_of_iter / 10) == 0:
            print(str((i - starting_loc) / num_of_iter * 100) + '%')

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
    alpha = 0.005
    alpha_decay = 1
    batch_size = 100
    num_of_steps = 600

    time1 = datetime.datetime.now()
    train_network(alpha, alpha_decay, batch_size, num_of_steps)
    time2 = datetime.datetime.now()

    print(time2 - time1)

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
                            # if i == 3:
                            #     values['weights'][i][j].append(np.random.randn() * np.sqrt(2 / (non[i] + non[i + 1])))
                            # else:
                                values['weights'][i][j].append(np.random.randn() * np.sqrt(2 / non[i]))

                if i != 0:
                    values['biases'].append([])

                    for j in range(non[i]):
                        values['biases'][i - 1].append(0.1)

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

        win = pygame.display.set_mode((520, 520))

        def draw_window(click, mouse, hold):
            global image, text_surface, drawing, erasing, drawing_thickness

            win.fill((0, 0, 0))

            for y in range(28):
                for x in range(28):
                    pygame.draw.rect(win, (max(20, image[y][x]), max(20, image[y][x]), max(20, image[y][x])), (x * 15, y * 15, 14, 14))

            pygame.draw.rect(win, (220, 30, 30), generate_answer)
            pygame.draw.rect(win, (30, 30, 220), generate_image_in_set)
            pygame.draw.rect(win, (30, 30, 30), drawing_thickness_slider)
            pygame.draw.rect(win, (50, 50, 50), drawing_thickness_rect)

            if drawing:
                pygame.draw.rect(win, enabled_drawing_color, enable_draw)
            else:
               pygame.draw.rect(win, disabled_drawing_color, enable_draw)

            if erasing:
                pygame.draw.rect(win, enabled_drawing_color, enable_erase)
            else:
               pygame.draw.rect(win, disabled_drawing_color, enable_erase)

            win.blit(text_surface, (195, 440))
            
            if click:
                if pygame.Rect.colliderect(generate_image_in_set, mouse):
                    rand_num = np.random.randint(0, 10000)
                    image = test_x[rand_num]

                elif pygame.Rect.colliderect(enable_draw, mouse):
                    drawing = not drawing
                    erasing = False

                elif pygame.Rect.colliderect(enable_erase, mouse):
                    erasing = not erasing
                    drawing = False

                elif pygame.Rect.colliderect(generate_answer, mouse):
                    result = multiply_matrixes(0, np.array(image).flatten() / 255)
                    highest_result = 0
                    answer = None

                    for i in range(len(result[len(result) - 1])):
                        if result[len(result) - 1][i] > highest_result:
                            highest_result = result[len(result) - 1][i]
                            answer = i

                    text_surface = font.render(str(answer), False, (230, 230, 230))

            if hold:
                if pygame.Rect.colliderect(pygame.Rect(0, 0, 420, 420), mouse) and drawing:
                    for y in range(len(image)):
                        for x in range(len(image[y])):
                            distance = np.sqrt(((x * 15 + 7.5) - mouse.x) ** 2 + ((y * 15 + 7.5) - mouse.y) ** 2)

                            if distance <= drawing_thickness * 15:
                                image[y][x] = max(255 - (distance / (drawing_thickness * 15) * 255), image[y][x])

                elif pygame.Rect.colliderect(pygame.Rect(0, 0, 420, 420), mouse) and erasing:
                    for y in range(len(image)):
                        for x in range(len(image[y])):
                            distance = np.sqrt(((x * 15 + 7.5) - mouse.x) ** 2 + ((y * 15 + 7.5) - mouse.y) ** 2)

                            if distance <= erasing_thickness * 15:
                                image[y][x] = 0

                elif pygame.Rect.colliderect(drawing_thickness_rect, mouse):
                    if mouse.x <= 440:
                        drawing_thickness_rect.x = 420
                    elif mouse.x >= 500:
                        drawing_thickness_rect.x = 480
                    else:
                        drawing_thickness_rect.x = mouse.x - 20

                    drawing_thickness = 1 + ((drawing_thickness_rect.x - 420) / 30)

            pygame.display.update()

        run = True
        clock = pygame.time.Clock()
        click = False
        drawing = False
        erasing = False
        hold = False
        drawing_thickness = 1.5
        erasing_thickness = 4

        font = pygame.font.SysFont('calibry', 50)
        small_font = pygame.font.SysFont('calibry', 20)
        text_surface = font.render('', False, (230, 230, 230))

        generate_image_in_set = pygame.Rect(0, 420, 100, 100)
        generate_answer = pygame.Rect(420, 420, 100, 100)

        enable_draw = pygame.Rect(420, 0, 100, 100)
        enable_erase = pygame.Rect(420, 100, 100, 100)
        drawing_thickness_slider = pygame.Rect(420, 230, 100, 30)
        drawing_thickness_rect = pygame.Rect(450, 225, 40, 40)

        enabled_drawing_color = (30, 220, 30)
        disabled_drawing_color = (60, 60, 60)

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
                    hold = True
                if event.type == pygame.MOUSEBUTTONUP:
                    hold = False

            pos = pygame.mouse.get_pos()
            mouse = pygame.Rect(pos[0], pos[1], 1, 1)
            draw_window(click, mouse, hold)

        pygame.quit()