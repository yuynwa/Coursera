import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = '../data/rock_paper_scissor/sign-language-mnist/'

train_data_dir = data_dir + 'sign_mnist_train.csv'
validation_data_dir = data_dir + 'sign_mnist_test.csv'


def get_data(filename):

    images = []
    labels = []

    headers = True
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=',')

        for row in csv_reader:
            if headers:
                headers = False
            else:
                labels.append(row[0])
                images.append(np.array_split(row[1:], 28))

    # with open(filename) as f:
    #
    #     for x in f.readlines():
    #
    #         if headers:
    #             headers = False
    #             continue
    #
    #         d = x.replace(' ', '').split(',')
    #         labels.append(d[0])
    #
    #         img = np.array_split(d[1:], indices_or_sections=28)
    #         images.append(img)

    images = np.array(images).astype(dtype=float)
    labels = np.array(labels).astype(dtype=float)

    return images, labels


training_images, training_labels = get_data(train_data_dir)
testing_images, testing_labels = get_data(validation_data_dir)

# training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
# testing_images = testing_images.reshape(testing_images.shape[0], 28, 28, 1)
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)


# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)



train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
)

train_generator = train_datagen.flow(
    x=training_images,
    y=training_labels,
    batch_size=32,
)


validation_datagen = ImageDataGenerator(
    rescale=1/255,
)

validation_generator = validation_datagen.flow(
    x=testing_images,
    y=testing_labels,
    batch_size=32,
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax'),
])


from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'],
)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=int(training_images.shape[0] / 32) + 1,
    epochs=10,
    validation_steps=int(testing_images.shape[0] / 32) + 1,
    validation_data=validation_generator,
)

model.evaluate(testing_images, testing_labels)


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and validation acccc')
plt.legend()
plt.figure()


plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()