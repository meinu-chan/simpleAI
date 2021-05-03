import tensorflow as tf
import numpy as  np
import os
import sys
import matplotlib.pyplot as plt

class_names = ['zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine']

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def trainModel():
    weights = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, verbose=1)

    model.fit(train_images, train_labels, epochs=10, callbacks=[weights])

    os.listdir(save_dir)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

def loadModel():
    model.load_weights(save_path)

    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Restored model, accuracy: {100 * acc}%")

def predict_digit(arr):
    arr = np.array(arr)

    arr = np.expand_dims(arr, 0)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predict = probability_model.predict(arr)

    print(class_names[np.argmax(predict)])

    plot_value_array(predict[0])
    plt.show()

def plot_value_array( predict):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predict, color="#777777")
  plt.ylim([0, 1])
  return

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

save_path = 'trainedModel/weight.ckpt'
save_dir = os.path.dirname(save_path)

try:
    loadModel()
except tf.errors.NotFoundError:
    trainModel()