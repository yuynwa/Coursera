#!usr/bin/env python
# -*- coding: utf-8 -*-



import tensorflow as tf

class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if logs.get('acc') > 0.90:
            print('\nReached 90% accuracy, so cancelling training!\n')

            self.model.stop_training = True



if __name__ == '__main__':


    mnist = tf.keras.datasets.fashion_mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train, X_test = X_train / 255.0, X_test / 255.0

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)



    model = tf.keras.models.Sequential([
        tf.layers.Conv2D(32, (2,2), activation='relu', input_shape=(28, 28, 1)),
        tf.layers.MaxPooling2D(2, 2),
        tf.layers.Conv2D(16, (2,2), activation='relu'),
        tf.layers.MaxPooling2D(2,2),
        tf.layers.Flatten(),
        tf.layers.Dense(units=512, activation='relu'),
        tf.layers.Dense(units=10, activation=tf.nn.softmax),
    ])


    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],

    )

    callbacks = [MyCallbacks()]
    model.fit(
        x = X_train,
        y = Y_train,
        epochs=30,
        callbacks=callbacks,
    )


    model.summary()


    test_loss = model.evaluate(
        x = X_test,
        y = Y_test,
    )

    print(test_loss)

    classifications = model.predict(X_test)

    print(Y_test[1234])
    print(classifications[1234])
