import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import os

from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator

pwd = os.getcwd()

path = pwd.replace(os.sep, "/")

TRAIN_PATH = path + "/Dataset/digit-recognizer/train.csv"
#
data = pd.read_csv(TRAIN_PATH)

N_LABELS = data["label"].nunique()


def parse_data(file_path):

    with open(file_path) as file:
        csv_reader = csv.reader(file, delimiter=",")

        images = []
        labels = []

        next(csv_reader)

        for row in csv_reader:
            label = row[0]
            image = row[1:]

            image = np.array(image).reshape((28, 28))

            images.append(image)
            labels.append(label)

        labels = np.array(labels).astype("float64")
        images = np.array(images).astype("float64")

    return images, labels


IMAGES, LABELS = parse_data(TRAIN_PATH)


def split_data(images, labels):

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.1,
                                                                          random_state=42)

    return train_images, val_images, train_labels, val_labels


TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS = split_data(IMAGES, LABELS)


def image_generator(train_images, val_images, train_labels, val_labels):

    train_images = np.expand_dims(train_images, axis=-1)
    val_images = np.expand_dims(val_images, axis=-1)

    train_data_generator = ImageDataGenerator(rescale=1/255.0,
                                              zoom_range=0.1,
                                              rotation_range=10,
                                              width_shift_range=0.1,
                                              height_shift_range=0.1,
                                              shear_range=0.2,
                                              fill_mode="nearest")

    train_generator = train_data_generator.flow(x=train_images,
                                                y=train_labels,
                                                batch_size=32)

    val_data_generator = ImageDataGenerator(rescale=1/255.0)

    val_generator = val_data_generator.flow(x=val_images,
                                            y=val_labels,
                                            batch_size=32)

    return train_generator, val_generator


TRAIN_GENERATOR, VAL_GENERATOR = image_generator(TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS)


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(126, activation="relu"),
        tf.keras.layers.Dense(N_LABELS, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


MODEL = create_model()

history = MODEL.fit(TRAIN_GENERATOR,
                    epochs=20,
                    validation_data=VAL_GENERATOR)

MODEL.save("saved_model/digit_recognizer.h5")

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
