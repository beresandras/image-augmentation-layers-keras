import augmentations
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

num_epochs = 10
batch_size = 64
width = 128

train_dataset = (
    tfds.load("cifar10", split="train", as_supervised=True, shuffle_files=True)
    .shuffle(10 * batch_size)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
test_dataset = (
    tfds.load("cifar10", split="test", as_supervised=True)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

model = keras.Sequential(
    [
        layers.Input(shape=(32, 32, 3)),
        preprocessing.Rescaling(1 / 255),
        augmentations.RandomColorJitter(),
        augmentations.RandomResizedCrop(),
        layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
