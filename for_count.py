from glob import glob 
import os

no_files = []
vul_files = []
files = glob('dataset/train_devign_for/*.json')
for file in files:
    if 'no-vul' in file:
        no_files.append(file)
    else:
        vul_files.append(file)
        
        
print(len(no_files))
print(len(vul_files))