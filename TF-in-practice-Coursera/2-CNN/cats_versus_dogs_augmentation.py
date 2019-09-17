import os, zipfile, random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

PET_IMAGES_DIR = '../data/tmp/PetImages'
CATS_V_DOGS_DIR = '../data/tmp/cats-v-dogs'

print(len(os.listdir(PET_IMAGES_DIR + '/Cat')))
print(len(os.listdir(PET_IMAGES_DIR + '/Dog')))

try:
    os.mkdir(CATS_V_DOGS_DIR)
    os.mkdir(CATS_V_DOGS_DIR + '/training')
    os.mkdir(CATS_V_DOGS_DIR + '/training/dogs')
    os.mkdir(CATS_V_DOGS_DIR + '/training/cats')

    os.mkdir(CATS_V_DOGS_DIR + '/testing')
    os.mkdir(CATS_V_DOGS_DIR + '/testing/dogs')
    os.mkdir(CATS_V_DOGS_DIR + '/testing/cats')
except OSError as e:
    print('error is:', e)



def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):


  files = []
  for filename in os.listdir(SOURCE):
      file = SOURCE + filename
      if os.path.getsize(file) > 0:
          files.append(filename)
      else:
          print(filename + " is zero length, so ignoring.")

  training_length = int(len(files) * SPLIT_SIZE)
  testing_length = int(len(files) - training_length)
  shuffled_set = random.sample(files, len(files))
  training_set = shuffled_set[0:training_length]
  testing_set = shuffled_set[-testing_length:]

  for filename in training_set:
      this_file = SOURCE + filename
      destination = TRAINING + filename
      copyfile(this_file, destination)

  for filename in testing_set:
      this_file = SOURCE + filename
      destination = TESTING + filename
      copyfile(this_file, destination)
# YOUR CODE ENDS HERE


CAT_SOURCE_DIR = "../data/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "../data/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "../data/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "../data/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "../data/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "../data/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


print(len(os.listdir('../data/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('../data/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('../data/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('../data/tmp/cats-v-dogs/testing/dogs/')))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])


model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['acc'],

)

TRAINING_DIR = '../data/tmp/cats-v-dogs/training/'
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

train_generator = train_datagen.flow_from_directory(
                    TRAINING_DIR,
                    batch_size=100,
                    class_mode='binary',
                    target_size=(150,150)
                )


VALIDATION_DIR = '../data/tmp/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(rescale=1.0/255.0,
                    rotation_range=40,
                    height_shift_range=0.2,
                    width_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )

validation_generator = train_datagen.flow_from_directory(
                    VALIDATION_DIR,
                    batch_size=100,
                    class_mode='binary',
                    target_size=(150,150)
                  )



history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)



# %matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

print(history.history)

print('===============')
print(history)
# import numpy as np
# from google.colab import files
# from keras.preprocessing import image
#
# uploaded = files.upload()
#
# for fn in uploaded.keys():
#
#     # predicting images
#     path = '/content/' + fn
#     img = image.load_img(path, target_size=(150, 150))
#         x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0] > 0.5:
#         print(fn + " is a dog")
#     else:
#         print(fn + " is a cat")
