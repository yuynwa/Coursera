import tensorflow as tf

# YOUR CODE SHOULD START HERE
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if logs['acc'] > 0.99:
            self.model.stop_training = True
            print("Reached 99% accuracy so cancelling training!")
        print(epoch, logs)

# YOUR CODE SHOULD END HERE


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# YOUR CODE SHOULD START HERE
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0

# YOUR CODE SHOULD END HERE
model = tf.keras.models.Sequential([
    # YOUR CODE SHOULD START HERE
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    # YOUR CODE SHOULD END HERE
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# YOUR CODE SHOULD START HERE
model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    batch_size=32,
    callbacks=[Callback()],
)

loss, acc = model.evaluate(
    x=x_test,
    y=y_test,

)

print(loss, acc)
# YOUR CODE SHOULD END HERE