#!usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt




if __name__ == '__main__':

    import tensorflow as tf


    # YOUR CODE STARTS HERE
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(epoch, logs)

            if logs.get('acc') > 0.90:
                self.model.stop_training = True
                print("Reached 99.8% accuracy so cancelling training!")


    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # YOUR CODE STARTS HERE
    training_images, test_images = training_images / 255.0, test_images / 255.0
    training_images = training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    print(training_images.shape)

    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        # YOUR CODE STARTS HERE
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
        # YOUR CODE ENDS HERE
    ])

    # YOUR CODE STARTS HERE
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        metrics=['accuracy'],
        loss='sparse_categorical_crossentropy',
    )

    callbacks = [MyCallback()]
    model.fit(
        x=training_images,
        y=training_labels,
        epochs=20,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(
        x=test_images,
        y=test_labels,
    )

    print(test_loss, test_acc)
    # classifications = model.predict(test_images)
    # print(classifications[432])
    # print(test_labels[432])
    # YOUR CODE ENDS HERE


