import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import csv

from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator


def parse_data(train_path):

    with open(train_path) as file:

        labels = []
        images = []

        csv_reader = csv.reader(file)

        next(csv_reader)

        for row in csv_reader:
            label = row[0]
            pixels = row[1:]

            image = np.array(pixels).reshape((28, 28))

            labels.append(label)
            images.append(image)

        labels = np.array(labels).astype("float64")
        images = np.array(images).astype("float64")

    return images, labels


def split_data(images, labels, test_size):

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size,
                                                                          random_state=42)

    return train_images, val_images, train_labels, val_labels


def data_generator(train_images, val_images, train_labels, val_labels):

    train_images = np.expand_dims(train_images, axis=-1)
    val_images = np.expand_dims(val_images, axis=-1)

    train_generator = ImageDataGenerator(rescale=1/255.,
                                         shear_range=0.4,
                                         rotation_range=15,
                                         zoom_range=0.2,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         fill_mode="nearest")

    train_gen = train_generator.flow(x=train_images,
                                     y=train_labels,
                                     batch_size=32)

    val_generator = ImageDataGenerator(rescale=1/255.)

    val_gen = val_generator.flow(x=val_images,
                                 y=val_labels,
                                 batch_size=32)

    return train_gen, val_gen


def create_model(n_labels):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(126, activation="relu"),
        tf.keras.layers.Dense(n_labels, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy" == 1.00):
            print(f"\n Training accuracy hit 100% so training stopped after {epoch} epochs! \n")

            self.model.stop_training = True


def model_training(model, train_gen, val_gen):

    callback = MyCallback()

    history = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=[callback])

    return history


def plot_graphs(history):

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs, acc, "red", label="Training Accuracy")
    plt.plot(epochs, val_acc, "blue", label="Validation Accuracy")
    plt.title("Training and Validation Accuracies")

    plt.legend()
    plt.show()

    plt.plot(epochs, loss, "red", label="Training Loss")
    plt.plot(epochs, val_loss, "blue", label="Validation Loss")
    plt.title("Training and Validation Losses")

    plt.legend()
    plt.show()


def main():

    pwd = os.getcwd()

    path = pwd.replace(os.sep, "/")

    train_path = path + "/Dataset/digit-recognizer/train.csv"

    images, labels = parse_data(train_path)

    n_labels = len(labels)
    test_size = 0.1

    train_images, val_images, train_labels, val_labels = split_data(images, labels, test_size)

    train_gen, val_gen = data_generator(train_images, val_images, train_labels, val_labels)

    model = create_model(n_labels)

    history = model_training(model, train_gen, val_gen)

    plot_graphs(history)

    model.save("saved_model/digit_recognizer.h5")


if __name__ == "__main__":
    main()
