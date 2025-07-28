import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.__version__
fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
index = 40

np.set_printoptions(linewidth=320)

print(f"label: {labels[training_labels[index]]}")
print(f'\nImage pixel array:\n {training_images[index]}')

plt.imshow(training_images[index])
training_images = training_images/255.0
test_images = test_images/255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.SparseCategoricalCrossentropy)
model.fit(training_images, training_labels, epochs=5)
