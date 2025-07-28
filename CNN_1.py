from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

train_dir = '/content/drive/MyDrive/dataset_sampah' # Correct path to your dataset
datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),                     # Input layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),        # Convolution layer
    tf.keras.layers.MaxPooling2D(2,2),                           # Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),        # Conv layer lagi
    tf.keras.layers.MaxPooling2D(2,2),                           # Pooling
    tf.keras.layers.Flatten(),                                  # Ubah 2D ke 1D
    tf.keras.layers.Dense(64, activation='relu'),               # Hidden layer
    tf.keras.layers.Dense(3, activation='softmax')              # Output: 3 kelas
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.show()

img = image.load_img('/content/drive/MyDrive/dataset_sampah/logam/R_1623.jpg', target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
class_names = list(train_data.class_indices.keys())
print("Prediksi:", class_names[np.argmax(prediction)])

