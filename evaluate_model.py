# Load the tensorflow model and evaluate it on the test set
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the test data
data = np.load('data-big-L2.npz')
data_x_test = data['data_x_test']
data_y_test = data['data_y_test']
print(data_x_test[0])

# labels = ["alarm clock", "book", "campfire", "cloud", "airplane", "umbrella"]
# labels = ["alarm clock", "book", "campfire", "cloud", "airplane", "spider", "key", "hamburger"]
# labels = ["alarm clock", "book", "campfire", "cloud", "airplane", "spider", "key", "hamburger", 'banana', "truck"]
# labels = ["ant", "axe", "bear", "owl", "squirrel", "tent"]
# labels = ["ant", "axe", "binoculars", "butterfly", "giraffe", "tent"]
labels = ["submarine", "speedboat", "rainbow", "star", "shark", "sea turtle", "octopus"]
lstm_model = load_model('C:/Users/fishm/Desktop/Code/VideoGame/SemesterProject/model-015.h5')

count = 0
unlikely = 0
samples = 1000
for i in range(samples):
    # Now load the first sample and run it through the model
    # print(data_x_test[i:i+1])
    result = lstm_model.predict(data_x_test[i:i+1])

    # Now print the true label and the predicted label
    # print(i)
    # print("True label:", labels[np.argmax(data_y_test[i])])
    # # print("Predicted label:", labels[np.argmax(result)])
    # print(labels)
    # print(np.around(result,3))
    real_class = np.argmax(data_y_test[i])
    if(real_class != np.argmax(result[0])):
        # print(labels[real_class])
        # print(result[0][real_class])
        count += 1
    if(result[0][real_class] < 0.33):
        unlikely += 1

print("________________________")
print(count/samples)
print(unlikely/samples)