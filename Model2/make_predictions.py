import pandas as pd
import numpy as np
import tensorflow as tf

import os
import csv


def parse_data(test_path):

    with open(test_path) as file:

        images = []

        csv_reader = csv.reader(file)

        next(csv_reader)

        for row in csv_reader:

            image = row[0:]

            pixels = []

            for pixel in image:

                pixel = float(pixel)
                pixels.append(pixel)

            image = np.array(pixels).reshape((28, 28, 1)) / 255.

            images.append(image)

        images = np.array(images).astype("float64")

        images = np.expand_dims(images, axis=0)

    return images


def main():
    pwd = os.getcwd()

    path = pwd.replace(os.sep, "/")

    test_path = path + "/Dataset/digit-recognizer/test.csv"

    sample_path = path + "/Dataset/digit-recognizer/sample_submission.csv"

    model_path = path + "/saved_model/digit_recognizer.h5"

    images = parse_data(test_path)

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(images[0])

    predictions = np.argmax(predictions, axis=1)

    sample_submission = pd.read_csv(sample_path)

    sample_submission["Label"] = predictions

    sample_submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
