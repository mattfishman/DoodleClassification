from read_data import create_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Convolution1D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# labels = ["alarm clock", "book", "campfire", "cloud"]
# data_x, data_y = create_data(labels)
# Load the data from the following save np.savez('data.npz', data_x=data_x, data_y=data_y)
data = np.load('data-big-L2.npz')
data_x = data['data_x']
data_y = data['data_y']
samples = data_x.shape[0]
stroke_length = data_x.shape[1]
classes = data_y.shape[1]

print(data_x.shape)
print(data_x[0])
# (4000, 196, 3)

print(data_y.shape)
print(data_y[0])
print(data_y[-1])
# (4000, 4)

# # The model should have three 1 dimensional convolutional layers, two LSTM layers with dropout, and a dense layer
# model = Sequential()
# # Start with the first 1 dimensional convolutional layer
# model.add(Convolution1D(filters=128, kernel_size=6, activation='relu', input_shape=(196, 3)))
# # Add a second 1 dimensional convolutional layer
# model.add(Convolution1D(filters=64, kernel_size=3, activation='relu'))
# # Add a third 1 dimensional convolutional layer
# model.add(Convolution1D(filters=1, kernel_size=1, activation='relu'))
# # Add two LSTM layers with dropout
# model.add(LSTM(units=128, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=128, return_sequences=False))
# model.add(Dropout(0.2))
# # Add a dense layer
# model.add(Dense(units=4, activation='softmax'))
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(stroke_length, 5)),
        tf.keras.layers.Conv1D(filters=48, kernel_size=5, activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
        tf.keras.layers.Conv1D(filters=96, kernel_size=3, activation='relu'),
        # tf.keras.layers.LSTM(units=128, return_sequences=True),
        # tf.keras.layers.Dropout(0.05),
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.LSTM(units=48, return_sequences=False),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=classes, activation='softmax')
    ]
)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')

history = model.fit(data_x, data_y, epochs=30, batch_size=stroke_length, validation_split=0.15, callbacks=[checkpoint], shuffle=True)

model.save('model.h5')