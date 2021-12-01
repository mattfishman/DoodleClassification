# notebook to convert raw quickdraw data using incremental RDP to fit everything within, say, 300 datapoints but not sacrifice quality.

import numpy as np
import util as util
import os
import json
import time
import random
from rdp import rdp
from tensorflow.keras import utils

np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

STROKE_COUNT = 100

def raw_to_lines(raw, epsilon=1.0):
    result = []
    N = len(raw)
    for i in range(N):
        line = []
        rawline = raw[i]
        M = len(rawline[0])
        if M <= 2:
            continue
        for j in range(M):
            line.append([rawline[0][j], rawline[1][j]])
        line = rdp(line, epsilon=epsilon) 
        result.append(line)
    return result

def parse_raw_line(file_path, max_count=20000):
    raw_file = open(file_path, 'r') 
    raw_lines = raw_file.readlines()
    num_drawings = len(raw_lines)
    all_strokes = []
    count = 1
    for i in range(num_drawings):
        if count > max_count:
            break
        raw_drawing = json.loads(raw_lines[i])['drawing']
        lines = raw_to_lines(raw_drawing)
        strokes = util.lines_to_strokes(lines)
        if i % 1000 == 0:
            print("i", i)
        if len(strokes) < 8:
            continue
        strokes[0, 0] = 0
        strokes[0, 1] = 0
        all_strokes.append(strokes)
        count += 1
    random.shuffle(all_strokes)
    print("strokes")
    print(len(all_strokes))
    return all_strokes

def further_simplify(temp_strokes, epsilon=0.5):
  temp_lines = util.strokes_to_lines(temp_strokes)
  new_lines = []
  for i in range(len(temp_lines)):
    temp_line = temp_lines[i]
    new_line = rdp(temp_line, epsilon)
    if len(new_line) > 2:
      new_lines.append(new_line)
  return util.lines_to_strokes(new_lines)

# all_strokes = parse_raw_line("C:/Users/fishm/Desktop/Code/VideoGame/SemesterProject/FishmanDoodleClassification/RawData/full_raw_tent.ndjson")
# length = len(all_strokes)
# NAME = "tent"

# dataLoader = util.DataLoader(all_strokes, batch_size=length)
# stroke3, stroke5, length = dataLoader.get_batch(0)
# stroke3 = np.array(stroke3)
# stroke5 = np.array(stroke5)
# print(stroke3.shape)
# print(stroke3[0])
# print(stroke5.shape)
# print(stroke5[0])
# print(length)

def create_data(labels):
    paths = []
    l = []
    data = []
    # data_x = np.empty((0,STROKE_COUNT,5), float)
    for label in labels:
        paths.append(f'C:/Users/fishm/Desktop/Code/VideoGame/SemesterProject/FishmanDoodleClassification/RawData/full_raw_{label}.ndjson')
    for path in paths:
        all_strokes = parse_raw_line(path)
        length = len(all_strokes)
        dataLoader = util.DataLoader(all_strokes, batch_size=length)
        stroke3, stroke5, length = dataLoader.get_batch(0)
        stroke5 = np.array(stroke5)
        l.append(stroke5.shape[0])
        data.append(stroke5)
    data_x = np.concatenate(data, axis=0)
    num_classes = len(labels)
    total = 0
    for length in l:
        total += length
    print("Length")
    print(l)
    data_y = np.zeros((1,total))
    c = 0
    index = 0
    for i in range(num_classes):
        data_y[:,index:index+l[c]] = c
        index += l[c]
        c += 1
    data_y = data_y.reshape(index,)
    data_y = utils.to_categorical(data_y)
    return data_x, data_y

def read_raw_input(raw_drawing):
    all_strokes = []
    lines = raw_to_lines(raw_drawing)
    strokes = util.lines_to_strokes(lines)
    strokes[0, 0] = 0
    strokes[0, 1] = 0
    all_strokes.append(strokes)
    dataLoader = util.DataLoader(all_strokes, batch_size=1)
    stroke3, stroke5, length = dataLoader.get_batch(0)
    data_x = np.array(stroke5)
    return data_x

if __name__ == '__main__':
    # labels = ["ant", "axe", "bear", "owl", "squirrel", "tent"]
    # labels = ["ant", "axe", "binoculars", "butterfly", "giraffe", "tent"]
    labels = ["submarine", "speedboat", "rainbow", "star", "shark", "sea turtle", "octopus"]
    data_x, data_y = create_data(labels)
    
    print(data_x.shape)
    print(data_y.shape)
    print(data_x[0])
    print(data_y[0])

    shuffler = np.random.permutation(len(data_x))
    data_x_shuffled = data_x[shuffler]
    data_y_shuffled = data_y[shuffler]
    split = (int)(len(data_x)*0.9)
    print(data_y_shuffled[0:5])

    np.savez('data-big-L2.npz', data_x=data_x_shuffled[:split], data_x_test=data_x_shuffled[split:], data_y=data_y_shuffled[:split], data_y_test=data_y_shuffled[split:])