#!usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt




if __name__ == '__main__':

    f, axarr = plt.subplots(3, 4)

    # print(f)
    # print(axarr)

    FIRST_IMG = 0
    SECOND_IMG = 7
    THIRD_IMG = 26


    CONVOLUTION_NUMBER = 1

    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_test, Y_test, epochs=1)
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(test_acc)


    # layer_outputs = [layer.output for layer in model.layers]
    #
    # activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    #
    #
    # for x in range(0, 4):
    #
    #     f1 = activation_model.predict(X_test[FIRST_IMG].reshape(1, 28, 28, 1))[x]
    #     axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    #     axarr[0, x].grid(False)
    #
    #     f2 = activation_model.predict(X_test[SECOND_IMG].reshape(1,28,28,1))[x]
    #     axarr[1, x].imshow(f2[0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    #     axarr[1, x].grid(False)
    #
    #     f3 = activation_model.predict(X_test[THIRD_IMG].reshape(1,28,28,1))[x]
    #     axarr[2, x].imshow(f3[0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    #     axarr[2, x].grid(False)
    #
    # plt.show()



