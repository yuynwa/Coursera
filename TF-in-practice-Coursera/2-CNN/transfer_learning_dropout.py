from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



local_weight_file = '../data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


pre_trained_model = InceptionV3(
    include_top=False,
    input_shape=(150, 150, 3),
    weights = None,
)

pre_trained_model.load_weights(filepath=local_weight_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_layer_output = last_layer.output

print(last_layer.output_shape)


x = layers.Flatten()(last_layer_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

print(pre_trained_model.input)
model = Model(
    pre_trained_model.input,
    x,
)

print(x)

from tensorflow.keras.optimizers import RMSprop

model.compile(
    optimizer=RMSprop(lr=0.0001),
    loss='binary_crossentropy',
    metrics=['acc'],
)


import zipfile, os

zip_dir = '../data/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(zip_dir, 'r')
zip_ref.extractall('../data/tmp')
zip_ref.close()

train_dir = '../data/tmp/cats_and_dogs_filtered/train'
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

val_dir = '../data/tmp/cats_and_dogs_filtered/validation'
val_cats_dir = os.path.join(val_dir, 'cats')
val_dogs_dir = os.path.join(val_dir, 'dogs')


train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150),
)


val_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
)

val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20,
)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=100,
    epochs=5,
    verbose=2,
    validation_data=val_generator,
    validation_steps=50,
)


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(len(acc))


plt.plot(epoch, acc, 'r', 'Train accuracy')
plt.plot(epoch, val_acc, 'b', 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
