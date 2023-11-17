# neural_network
my first simple neural network

I have finally made one file that operates every single neural network.
Plus, I have added covolutional neural network.
I have generalized the program, so that you can input as many neurons and as many layers as you want.
Cnn have to have fixed size (2 layers), because currently I am performing max pooling after each convolutional layer and it would break. One would technically work, but I haven't implemented logic for that.
My neural networks are in a folder called successful_networks.
Best one yet is 4-8-pure_cnn - 0.0005.json (roughly 97.6% accuracy).
I have also created better visualization for neural networks where you can see first layer weights or in case of cnn in kernels in any layer.