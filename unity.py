# # This program is going to allow our neural network to interface with Unity
# import numpy as np
# import tensorflow as tf

# # from mlagents_envs import UnityEnvironment
# # from mlagents_envs.exception import UnityEnvironmentException

# from mlagents_envs.environment import UnityEnvironment
# # This is a non-blocking call that only loads the environment.
# env = UnityEnvironment(file_name="3DBall", seed=1, side_channels=[])
# # Start interacting with the environment.
# env.reset()
# behavior_names = env.behavior_specs.keys()

# # # Now read the data from the unity environment
# # # This is the data that we are going to to predict
# # env = UnityEnvironment()
# # x_data = env

import zmq
import numpy as np
from tensorflow.keras.models import load_model
from simplify import read_raw_input

model_1 = load_model('C:/Users/fishm/Desktop/Code/VideoGame/SemesterProject/ModelBackup-L1/model-015.h5')
model_2 = load_model('C:/Users/fishm/Desktop/Code/VideoGame/SemesterProject/ModelBackup-L2/model-015.h5')

# labels = ["alarm clock", "book", "campfire", "cloud", "airplane", "spider", "key", "hamburger", "banana", "truck"]
labels_1 = ["ant", "axe", "binoculars", "butterfly", "giraffe", "tent"]
labels_2 = ["submarine", "speedboat", "rainbow", "star", "shark", "sea turtle", "octopus"]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    bytes_received = socket.recv(6000)
    array_received = np.frombuffer(bytes_received, dtype=np.float32).reshape(500,3)
    # print(array_received)
    # print(array_received[:,2])
    raw_drawing = []
    sub_array_x = []
    sub_array_y = []
    level = -1 * array_received[-1][2]
    for i in range(500):
        indicator = array_received[i][2]
        if(indicator == 0):
            sub_array_x.append(array_received[i][0])
            sub_array_y.append(array_received[i][1])
        elif(indicator == 1):
            if(len(sub_array_x) > 0):
                raw_drawing.append([sub_array_x, sub_array_y])
                sub_array_x = []
                sub_array_y = []
            sub_array_x.append(array_received[i][0])
            sub_array_y.append(array_received[i][1])
        else:
            if(len(sub_array_x) > 0):
                raw_drawing.append([sub_array_x, sub_array_y])
            break
    
    #print(raw_drawing)

    data_x = read_raw_input(raw_drawing)

    # print(data_x.shape)
    # print(data_x)
    if(level == 1):
        pred = model_1.predict(data_x)
        pred_index = np.argmax(pred)
        pred_label = labels_1[pred_index]
        print("\n_______________")
        print(level)
        print(pred_label)
        print(labels_1)
        print(np.around(pred,3))
    elif(level == 2):
        pred = model_2.predict(data_x)
        pred_index = np.argmax(pred)
        pred_label = labels_2[pred_index]
        print("\n_______________")
        print(level)
        print(pred_label)
        print(labels_2)
        print(np.around(pred,3))
    
    
    # print(pred)
    # print the max value of the prediction
    # print(np.amax(pred))

    bytes_to_send = pred.tobytes()
    # bytes_to_send = np.array([1, 0, 0, 0]).tobytes()
    socket.send(bytes_to_send)


# [[[183,178,174,169,162,156,149,142,135,130,125,120,116,112,106,101,96,91,84,79,74,69,63,57,51,46,52,57,62,67,72,76,80,85,92,97,103,109,114,118,123,129,133,128,123,115,110,102,97,91,83,75,67,60,52,46,41,36,27,20,16,15,15,15,15,15,17,23,28,33,38,44,50,58,63,69,75,82,91,97,102,107,113,119,124,130,136,143,151,160,165,170,177,182,187,193,202,207,215,221,226,232,239,245,253,259,265,271,272,272,272,271,268,264,259,252,244,232,223,214,205,198,192,187,182,181],[142,147,152,157,164,170,177,183,188,192,197,203,208,213,220,225,230,234,235,233,232,232,232,232,232,232,225,221,218,213,209,204,197,190,181,174,165,158,151,146,140,132,126,124,124,125,128,132,135,139,142,147,151,153,157,159,160,160,161,161,155,145,136,129,123,118,112,110,108,105,102,99,97,93,91,89,86,84,83,82,81,80,79,79,79,79,79,79,79,79,79,79,79,78,78,78,78,78,78,78,78,77,76,75,75,75,75,76,86,99,104,111,118,123,124,124,125,127,128,129,129,129,128,126,123,122],[0,20,34,50,67,84,106,124,134,150,168,185,202,217,252,301,371,417,885,953,985,1001,1018,1034,1051,1067,1184,1200,1218,1234,1251,1267,1285,1302,1318,1334,1352,1367,1384,1401,1418,1452,1488,1835,1850,1885,1918,2019,2036,2051,2068,2085,2101,2119,2134,2152,2167,2184,2217,2252,2318,2351,2367,2384,2401,2418,2454,2549,2567,2584,2600,2617,2634,2667,2703,2734,2767,2802,2834,2869,2902,2935,2967,3002,3034,3067,3100,3134,3167,3202,3219,3236,3268,3300,3334,3367,3403,3419,3453,3486,3517,3570,3901,3918,3933,3950,3967,4000,4083,4117,4155,4235,4268,4318,4335,4351,4366,4385,4401,4418,4435,4454,4471,4485,4517,4617]],[[224,224,224,224,224,225,227,230,237,244,250,255,255],[76,81,91,96,105,110,116,121,130,135,137,137,137],[5091,5118,5155,5169,5200,5218,5235,5253,5285,5318,5368,5455,5470]],[[135,133,132,130,126,123,117,111,106,101,97,91,86,81,73,64,59,54,58,61,65,70,75,81,87,93,98,103,108,109],[74,69,62,55,47,42,34,27,22,16,11,4,1,1,1,2,4,6,13,19,24,32,39,45,52,58,63,68,71,72],[6066,6204,6233,6267,6303,6318,6353,6384,6417,6469,6504,6552,6987,7118,7150,7183,7200,7250,7451,7466,7483,7500,7516,7533,7550,7567,7584,7600,7634,7683]]]