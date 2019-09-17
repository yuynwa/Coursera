#!usr/bin/env python
# -*- codning: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.preprocessing.image \
    import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1/255.0)


train_dir = './data'

ret = train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
)

print(ret)