#!usr/bin/env python
# -*- coding: utf-8 -*-


from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import zipfile, os

if __name__ == '__main__':


    if not os.path.exists('./data/catsanddogs'):

        local_zip = './data/kagglecatsanddogs_3367a.zip'
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('./data/catsanddogs')
        zip_ref.close()


    train_dogs_dir = os.path.join('./data/catsanddogs/PetImages/Dog')
    train_cats_dir = os.path.join('./data/catsanddogs/PetImages/Cat')

    dogs_names = os.listdir('./data/catsanddogs/PetImages/Dog')
    cats_names = os.listdir('./data/catsanddogs/PetImages/Cat')
    print(len(cats_names))


    print(cats_names)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D((2,2), (2,2)),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])


    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    train_datagen = ImageDataGenerator(rescale=1/255.0)
    train_generator = train_datagen.flow_from_directory(
        directory='./data/catsanddogs/PetImages',
        target_size=(100,100),
        batch_size=128,
        class_mode='binary',
        # directory,
        # target_size=(256, 256),
        # color_mode='rgb',
        # classes=None,
        # class_mode='categorical',
        # batch_size=32,
        # shuffle=True,
        # seed=None,
        # save_to_dir=None,
        # save_prefix='',
        # save_format='png',
        # follow_links=False,
        # subset=None,
        # interpolation='nearest'):
    )
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=98,
        epochs=2,
        verbose=1,
    )