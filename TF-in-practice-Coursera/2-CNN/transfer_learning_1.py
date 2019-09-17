import os
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import  InceptionV3

local_weight_file = '../data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None,
)

pre_trained_model.load_weights(
    filepath=local_weight_file,
)




for layer in pre_trained_model.layers:
    layer.trainable = False
    print(layer)

last_layer = pre_trained_model.get_layer('mixed7')

print(last_layer.output_shape)
print(last_layer.output)


last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = Model(
    pre_trained_model.input,
    x,
)

model.compile(
    optimizer=RMSprop(lr=0.0001),
    loss='binary_crossentropy',
    metrics=['acc'],
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

local_zip = '../data/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(file=local_zip, mode='r')

zip_ref.extractall('../data/tmp')
zip_ref.close()

base_dir = '../data/tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    height_shift_range=0.2,
    width_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
)


train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150),
)


validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator =  validation_datagen.flow_from_directory(
    directory=validation_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150),
)


history = model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=2,
    validation_steps=50,
    verbose=2,
)


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
