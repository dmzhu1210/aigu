import os
from glob import glob
from utils.tools import *
from tqdm import tqdm

def generation(path):
    with open(path) as f:
        content = f.read()
    codes = pre_word2vec(content)
    return codes

if __name__ == '__main__':
    root = 'dataset/reveal_c'
    for folder in os.listdir('dataset/reveal_c'):
        files = glob(os.path.join(root, folder, '*.c'))
        no_vul = []
        vul = []
        for file in tqdm(files):
            label = 0 if 'novul' in file else 1
            basename = os.path.basename(file)
            spe_name = basename.split('.')[0]
            feature = generation(file)
            if feature is None:
                continue
            torch.save(feature, os.path.join('dataset/reveal_pt', basename.replace('.c', '.pt')))