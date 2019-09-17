import tensorflow as tf
import numpy as np
from tensorflow import keras


model = keras.Sequential(layers=[
    keras.layers.Dense(units=1, input_shape=[1],)
])


# model.compile(
#     optimizer='adam',
#     loss='mean_squared_error',
#     metrics=['acc'],
# )
model.compile(
    optimizer='sgd',
    loss='mean_squared_error',
)

xs = np.array([-1, 0, 1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,    8.0], dtype=float)
ys = np.array([0, 0,  100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0], dtype=float)

model.fit(
    x=xs,
    y=ys,
    epochs=300,
)

print(model.predict([20]))