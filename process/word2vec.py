import argparse
from gensim.models import Word2Vec
import json
import re
import os
import nltk

def remove_commit(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    #subtoken
    return code

def convert(name, nolower=False):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    if nolower:
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    else:
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def my_tokenizer(code, nolower=False):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    ## Remove newlines & tabs
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(/)|(\\r)','',code)
    ## Mix split (characters and words)
    splitter = '\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)|(/)'
    code = re.split(splitter,code)
    ## Remove None type
    code = list(filter(None, code))
    code = list(filter(str.strip, code))
    # snakecase -> camelcase and split camelcase
    code_1 = []
    for i in code:
        code_1 += convert(i, nolower).split('_')
    #filt
    code_2 = []
    for i in code_1:
        if i in ['{', '}', ';', ':']:
            continue
        code_2.append(i)
    return(code_2)

def train(args):
    # files = args.data_paths
    from glob import glob
    from tqdm import tqdm
    files = []
    # for folder in os.listdir('diverse/c'):
    files = glob(os.path.join('diverse/c', '*.c'))
        # files = files + files_
    sentences = []
    for f in tqdm(files):
        data = []
        with open(f, 'r') as fr:
            # data = json.load(fr)
            # print(len(data))
            content = fr.read()
            #for line in fr.readlines():
            #    data.append(json.loads(line))
        # for e in range(len(data)):
            # sentences.append(my_tokenizer(data[e]['func']))
            sentences.append(my_tokenizer(content))
    wvmodel = Word2Vec(sentences, min_count=args.min_occ, workers=8, vector_size=args.embedding_size)
    print('Embedding Size : ', wvmodel.vector_size)
    for i in range(args.epochs):
        wvmodel.train(sentences, total_examples=len(sentences), epochs=1)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    print(args.save_model_dir+args.model_name)
    save_file_path = os.path.join(args.save_model_dir, args.model_name)
    wvmodel.save(save_file_path)

def check(args):
    files = args.data_paths
    data = []
    with(open(files[0], 'r')) as fr:
        for line in fr.readlines():
                data.append(json.loads(line))
    code = remove_commit(data[0]['code'])
    #code_ = [token.strip() for token in code.split()]
    code_ = nltk.word_tokenize(code)
    print(code_)
    
    
def load(args):
    model = Word2Vec.load('word2vec/diverse_train_subtoken_data')
    print(len(model.wv.key_to_index))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_paths', type=str, nargs='+', default=['data/code_train.json', 'data/code_test.json'])
    #parser.add_argument('--data_paths', type=str, nargs='+', default=['dataset/devign_full_data_with_slices'])
    parser.add_argument('--data_paths', type=str, nargs='+', default=['dataset.json'])
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-bin', '--save_model_dir', type=str, default='./')
    parser.add_argument('-n', '--model_name', type=str, default='word2vec/diverse_train_subtoken_data')
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-eb', '--embedding_size', type=int, default=100)
    parser.add_argument('-mp', '--model_path', type=str, default='fan_tokenizer')
    args = parser.parse_args()
    #check(args)
    # train(args)
    load(args)