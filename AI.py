import tensorflow as tf
import numpy as  np
import os
import sys
import threading
import matplotlib.pyplot as plt
# import datetime
from tensorboard.plugins.hparams import api as hp

class_names = ['zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine']

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

def trainModel():
    callbacks = tf.keras.callbacks

    weights = callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, verbose=1)

    tensorboard = callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

    # hparams_callback = hp.KerasCallback(log_dir, {
    # 'num_relu_units': 512,
    # 'dropout': 0.2
    # })

    model.fit(train_images, train_labels, epochs=5, callbacks=[weights, tensorboard])

    os.listdir(save_dir)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

def loadModel():
    model.load_weights(save_path)

    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Restored model, accuracy: {100 * acc}%")

def predict_digit(arr):
    arr = np.array(arr)/np.amax(arr)

    arr = np.expand_dims(arr, 0)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predict = probability_model.predict(arr)

    # print(predict[0])

    return class_names[np.argmax(predict)]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    # tf.keras.layers.Dense(16, activation=tf.nn.relu),
    # tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

save_path = 'trainedModel/weight.ckpt'
save_dir = os.path.dirname(save_path)
log_dir="logs/" 

try:
    loadModel()
except tf.errors.NotFoundError:
    trainModel()