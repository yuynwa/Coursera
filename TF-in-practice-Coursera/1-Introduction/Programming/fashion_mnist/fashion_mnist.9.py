#!usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import matplotlib.pyplot as plt



if __name__ == '__main__':

    mnist = tf.keras.datasets.fashion_mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # plt.imshow(X_train[314])
    # plt.show()

    X_train, X_test = X_train / 255.0, X_test / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3,3),
                               activation='relu',
                               input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),

        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(3,3),
                               activation='relu',
                               ),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        # tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        # tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
        # tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  )


    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):

            if logs.get('acc') > 0.950:
                print('\nReached 95% accuracy so cancelling training!\n')
                self.model.stop_training = True



    callbacks = [MyCallback()]
    model.fit(x=X_train,
              y=Y_train,
              epochs=1,
              callbacks=callbacks)


    model.evaluate(x=X_test,
                   y=Y_test)

    classifications = model.predict(X_test)

    print(classifications[2345])
    print(Y_test[2345])


