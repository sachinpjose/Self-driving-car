# Self-driving-car

The Project for training and testing the sslf driving car through the Udacity Car simulator.

Built and trained a convolutional neural network for end-to-end driving in a simulator, using TensorFlow and Keras. 
Used optimization techniques such as regularization and dropout to generalize the network for driving on multiple tracks.


Script Details
config.py : Configuration and HyperParameters for the model.
model.py : Script for defining and training the model.
load_data.py : Script for preprocessing and augmenting the image.
drive.py : Script for driving the autonomous car in simulator. 


Network architecture : 
Network architecture is borrowed from the aforementioned NVIDIA paper in which they tackle the same problem of steering angle prediction, just in a slightly more unconstrained environment.

Input normalization is implemented through a Lambda layer, which constitutes the first layer of the model. In this way input is standardized such that lie in the range [-1, 1]: of course this works as long as the frame fed to the network is in range [0, 255].

The choice of ELU activation function (instead of more traditional ReLU) come from this model of CommaAI, which is born for the same task of steering regression. On the contrary, the NVIDIA paper does not explicitly state which activation function they use.

Convolutional layers are followed by 3 fully-connected layers: finally, a last single neuron tries to regress the correct steering value from the features it receives from the previous layers.


Training Details
Model was compiled using Adam optimizer with default parameters and mean squared error loss w.r.t. the ground truth steering angle. Training is done using Google Colab.

Testing the model
After the training, the network can successfully drive on both tracks. Quite surprisingly, it drives better and smoother on the test track with respect to the training track. 