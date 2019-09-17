#!/use/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


    model.fit(
        x = training_images,
        y = training_labels,
    )


    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)

    print(classifications[0])
    print(test_labels[0])