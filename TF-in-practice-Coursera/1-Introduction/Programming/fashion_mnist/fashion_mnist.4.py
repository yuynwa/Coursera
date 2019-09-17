#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf



if __name__ == '__main__':

    print(tf.__version__)

    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
        # tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])


    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
    )

    model.fit(training_images, training_labels, epochs=30)
    model.evaluate(test_images, test_labels)

    classficatioins = model.predict(
        test_images,
    )

    print(classficatioins[0])
    print(test_labels[0])