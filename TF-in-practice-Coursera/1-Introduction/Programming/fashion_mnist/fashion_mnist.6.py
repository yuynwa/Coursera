#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nReached 60% accuracy so cancelling training!\n")
            self.model.stop_training = True

    # def on_batch_begin(self, batch, logs=None):


if __name__ == '__main__':

    print(tf.__version__)


    mnist = tf.keras.datasets.fashion_mnist

    (training_images, training_labels), (test_images, test_label) = mnist.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])


    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
    )


    callback = MyCallback()

    model.fit(
        training_images,
        training_labels,
        epochs=100,
        callbacks=[callback],
    )


    model.evaluate(
        test_images,
        test_label,
    )

    classifications = model.predict(test_images)

    print(classifications[123])
    print(test_label[123])