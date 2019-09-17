#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if logs.get('acc') > 0.950:
            print("\nReached 99% accuracy so cancelling training!\n")
            self.model.stop_training = True


if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train, X_test = X_train / 255.0, X_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=62, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)

    callbacks = [MyCallback()]
    model.fit(x=X_train,
              y=Y_train,
              epochs=10,
              callbacks=callbacks,)


    model.evaluate(x=X_test,
                   y=Y_test)


    classifications = model.predict(X_test)

    print(classifications[2342])
    print(Y_test[2342])

