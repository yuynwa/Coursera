#!/use/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras


model = keras.Sequential([keras.layers.Dense(
    units=1,
    input_shape=[1]
)])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)


Xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

Ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(
    x=Xs,
    y=Ys,
    epochs=500,
)


print(model.predict([10]))


