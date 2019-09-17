#!/usr/bin/env python
# -*- coding: utf-8 -*-



import tensorflow as tf



if __name__ == '__main__':


    mnist = tf.keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2], 1) / 255.0
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1) / 255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), name='conv_2d_1'),
        tf.keras.layers.MaxPooling2D((2,2), (2,2), name='max_pool_1'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='conv_2d_2'),
        tf.keras.layers.MaxPooling2D(2,2,name='max_pool_2'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


    model.fit(X_train, Y_train, epochs=1)

    test_loss = model.evaluate(X_test, Y_test)

    print(test_loss)

    print(model.summary())

    classifications = model.predict(X_test)

    print(classifications[3])
    print(Y_test[3])

