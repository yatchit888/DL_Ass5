from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=1):
    """Define a dense model with a single layer with the following parameters:
    input_length: the number of inputs
    activation_f: the activation function
    output_length: the number of outputs (number of neurons)"""
    model = keras.Sequential()
    model.add(layers.Dense(output_length, activation=activation_f, input_shape=(input_length,)))
    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         activation_func_array=['relu','sigmoid'],
                                         hidden_layer_size=10,
                                         output_length=1):
    """Define a dense model with a hidden layer with the following parameters:
    input_length: the number of inputs
    activation_func_array: the activation function for the hidden layer and the output layer
    hidden_layer_size: the number of neurons in the hidden layer
    output_length: the number of outputs (number of neurons in the output layer)"""

    model = keras.Sequential()
    model.add(layers.Dense(hidden_layer_size, activation=activation_func_array[0], input_shape=(input_length,)))
    model.add(layers.Dense(output_length, activation=activation_func_array[1]))
    return model



def get_mnist_data():
    """Get the MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255 
    return (x_train, y_train), (x_test, y_test)

from keras.utils import to_categorical

def fit_mnist_model(x_train, y_train, model, epochs=10, batch_size=32):
    """Fit the model to the data.
    compile the model and add parameters for  the "optimizer", the loss function, 
    and the metrics, Hint: use categorical crossentropy for the loss function .

    then fit the model on the training data. (pass the epochs and batch_size params)
    """
    # One-hot encode the target data
    y_train = to_categorical(y_train, num_classes=10)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
  
def evaluate_mnist_model(x_test, y_test, model):
    """Evaluate the model on the test data.
    Hint: use model.evaluate() to evaluate the model on the test data.
    """
    # One-hot encode the target data
    y_test = to_categorical(y_test, num_classes=10)

    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

