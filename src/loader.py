import os

import numpy as np
from PIL import Image


def load(path_to_image):
    img_matrix = np.asarray(Image.open(path_to_image))
    img_vector = []
    for row in img_matrix:
        for rgb_pixel in row:
            grayscale_pixel = abs((rgb_pixel[0] / 255) - 1)
            img_vector.append(grayscale_pixel)
    row_vector = np.asarray(img_vector, dtype=np.float64)
    return np.reshape(row_vector, (len(row_vector), 1))


def load_training_data():
    labeled_data = []
    for number in range(0, 9):
        path = "../" + str(number)
        label = label_for(number)
        names = [path + "/" + name for name in os.listdir(path)]
        for name in names:
            labeled_data.append((load(name), label))
    return labeled_data


def label_for(number):
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label[number] = 1
    row_vector = np.asarray(label, dtype=np.float64)
    return np.reshape(row_vector, (len(row_vector), 1))
