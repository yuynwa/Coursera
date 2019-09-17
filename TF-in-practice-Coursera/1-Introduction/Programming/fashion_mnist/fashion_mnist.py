#!/use/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt


"""
    
    This problem is from https://leetcode.com

    https://leetcode.com/problems/valid-parentheses/
    
"""

if __name__ == '__main__':

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    idx = 1
    plt.imshow(train_images[idx])
    print(train_labels[idx])
    print(train_images[idx])

    train_images = train_images / 255.0
    test_images = test_images / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),

    ])


    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


    model.fit(
        x=train_images,
        y=train_labels,
        epochs=5,
    )

    model.evaluate(x=test_images, y=test_labels)



    classifications = model.predict(test_images)

    print(classifications[0])

    print(test_labels[0])