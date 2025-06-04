import os
import glob
import random
import json

ratio = '8:1:1'

files1, files2 = list(), list()
for file in glob.glob('dataset/train_diverse/*.json'):
    if 'no-vul' in file or 'novul' in file:
        files1.append(file)
    else:
        files2.append(file)
        

random.shuffle(files1)
random.shuffle(files2)
files1 = files1
files2 = files2
print(len(files1))
print(len(files2))

train_len1 = int(len(files1)*0.8)
train_len2 = int(len(files2)*0.8)

train_data = files1[:train_len1] + files2[:train_len2]

files1 = files1[train_len1:]
files2 = files2[train_len2:]

valid_len1 = len(files1)//2
valid_len2 = len(files2)//2

valid_data = files1[:valid_len1] + files2[:valid_len2]

test_data  = files1[valid_len1:] + files2[valid_len2:]

all_data = []
for file in train_data:
    tmp = {
        'file_path' : file
    }
    all_data.append(tmp)
with open('train_diverse.json', 'w') as f:
    json.dump(all_data, f)
    
    
all_data = []
for file in valid_data:
    tmp = {
        'file_path' : file
    }
    all_data.append(tmp)
with open('valid_diverse.json', 'w') as f:
    json.dump(all_data, f)


all_data = []
for file in test_data:
    tmp = {
        'file_path' : file
    }
    all_data.append(tmp)
with open('test_diverse.json', 'w') as f:
    json.dump(all_data, f)
