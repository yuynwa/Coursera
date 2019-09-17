#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import keras


class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if logs.get('acc') > 0.85:

            print("\nReaches 85% accuracy so cancelling training!\n")
            self.model.stop_training = True



if __name__ == '__main__':

    mnist = keras.datasets.fashion_mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train, X_test = X_train / 255.0, X_test / 255.0


    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=256, activation=tf.nn.relu),
        keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [MyCallback()]
    model.fit(
        X_train,
        Y_train,
        epochs=2,
        callbacks=callbacks,
    )

    model.evaluate(
        X_test,
        Y_test,
    )


    classifications = model.predict(X_test)

    print(classifications[234])
    print(Y_test[234])