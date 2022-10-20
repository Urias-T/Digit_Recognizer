import numpy as np
import tensorflow as tf
import os
import csv
import pandas as pd

pwd = os.getcwd()

path = pwd.replace(os.sep, "/")

TEST_PATH = path + "/Dataset/digit-recognizer/test.csv"


def parse_data(file_path):

    with open(file_path) as file:
        csv_reader = csv.reader(file, delimiter=",")

        images = []

        next(csv_reader)

        for row in csv_reader:
            image = row[0:]

            pixels = []

            for pixel in image:

                floated_pixel = float(pixel)

                pixels.append(floated_pixel)

            image = np.array(pixels).reshape((28, 28, 1)) / 255.0

            images.append(image)

        images = np.array(images).astype("float64")

        images = np.expand_dims(images, axis=0)

    return images


IMAGES = parse_data(TEST_PATH)

MODEL_PATH = path + "/saved_model/digit_recognizer.h5"


MODEL = tf.keras.models.load_model(MODEL_PATH)

predictions = MODEL.predict(IMAGES[0])

predictions = np.argmax(predictions, axis=1)

sample_submission = pd.read_csv(path + "/Dataset/digit-recognizer/sample_submission.csv")

sample_submission["Label"] = predictions

sample_submission.to_csv("submission.csv", index=False)
