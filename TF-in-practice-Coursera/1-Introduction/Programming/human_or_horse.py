#!/use/bin/env python
# -*- coding: utf-8 -*-

import os
import zipfile
# from google.colab import files



local_zip = './data/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./data/horse-or-human')
zip_ref.close()


train_horse_dir = os.path.join('./data/horse-or-human/horses')
train_human_dir = os.path.join('./data/horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir('./data/horse-or-human/humans')
print(train_human_names[:10])


print("total training horse images: ", len(os.listdir(train_horse_dir)))
print("total training human images: ", len(os.listdir(train_human_dir)))


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4
pic_idx = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
pic_idx += 8

next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_idx - 8: pic_idx]]
print(next_horse_pix)

next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_idx - 8: pic_idx]]
print(next_human_pix)

# for idx, path in enumerate(next_horse_pix + next_human_pix):
#
#     sp = plt.subplot(nrows, ncols, idx + 1)
#     sp.axis('Off')
#
#     img = mpimg.imread(path)
#     plt.imshow(img)
#
# plt.show()


import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,300, 3), name='conv2d_1'),
    tf.keras.layers.MaxPooling2D((2,2), padding='same', name='maxpooling_1'),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2d_2'),
    tf.keras.layers.MaxPooling2D((2,2), padding='same', name='maxpooling_2'),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2d_3'),
    tf.keras.layers.MaxPooling2D((2,2), padding='same', name='maxpooling_3'),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2d_4'),
    tf.keras.layers.MaxPooling2D((2,2), padding='same', name='maxpooling_4'),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2d_5'),
    tf.keras.layers.MaxPooling2D((2,2), padding='same', name='maxpooling_5'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation=tf.nn.softmax),

])



model.summary()


from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(0.001),
    metrics=['accuracy'],
    loss='binary_crossentropy',
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dataGen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_dataGen.flow_from_directory(
    './data/horse-or-human',
    target_size=(300,300),
    batch_size=128,
    class_mode='binary',
)


model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=2,
    # epochs=15,
    verbose=1
)

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
#     img = image.load_img(path, target_size=(300, 300))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0] > 0.5:
#         print(fn + " is a human")
#     else:
#         print(fn + " is a horse")
#


import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

successive_outputs = [layer.output for layer in model.layers]

visualization_model = tf.keras.models.Model(inputs=model.input, outputs = successive_outputs)

horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]

img_path = random.choice(horse_img_files = human_img_files)

img = load_img(img_path,
               target_size=(300, 300))
x = img_to_array(img=img)
x = x.reshape((1,) + x.shape)


x /= 255.0

successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# Clean Up
# Before running the next exercise,
# run the following cell to terminate the-
# kernel and free memory resources:

# import os, signal
# os.kill(os.getpid(), signal.SIGKILL)
