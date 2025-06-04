import os
import re
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from word2vec import my_tokenizer
from gensim.models import Word2Vec
from transformers import RobertaModel, RobertaTokenizer

codep = r'\((.*?)\)[^)]*$'
nodep = r'"(\d+)" \[label = <.*?<SUB>(\d+)</SUB>>'
nodep2 =  r'"(\d+)" \[label ='
edgep = r'"(\d+)"\s*->\s*"(\d+)"'
atobp = r'"(\d+)"\s*->\s*"(\d+)"'
labep = r'label\s*=\s*"([^"]*)"'
# model_path = 'devign_dataset/devign_wv_models/devign_train_subtoken_data'
# model_path = '/root/autodl-tmp/word2vec/reveal2_train_subtoken_data'
# model = Word2Vec.load(model_path)
# model_path = 'word2vec_p/devign_train_subtoken_data'
model_path = 'word2vec/diverse_train_subtoken_data'

codebert_tokenizer = RobertaTokenizer.from_pretrained("pretrained/codebert/")
code_model = RobertaModel.from_pretrained("pretrained/codebert", output_hidden_states=True)
# code_model = code_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_model.to(device)
    
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.code = RobertaModel.from_pretrained("pretrained/codebert", output_hidden_states=True)
        self.code.pooler = nn.Sequential(nn.Linear(in_features=768, out_features=200, bias=True), nn.Tanh())
        # self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(200, 2)
    def forward(self, x, y):
        # pre = self.code(x, y)[0][:, 0]
        # final = torch.sum(ouputs.last_hidden_state, dim=1)
        pre = self.code(x, y)
        pre = pre.pooler_output[:, 0, :]
        # pre = self.drop(pre)
        # pre = self.linear(pre)
        return pre
    
# model_path = 'tree_20.bin'
# code_model = mymodel()
# code_model.load_state_dict(torch.load(model_path))
# code_model.to(device)
# code_model.eval()

def encode_input(input):
    """
    Encodes the input text using the provided tokenizer.
    :param text: The input text to be encoded.
    :param tokenizer: The tokenizer object.
    :return: Tuple containing input_ids and attention_mask tensors.
    """
    return input['input_ids'], input['attention_mask']

operators = [
    '=',
    "++",     # 自增运算符
    "--",     # 自减运算符
    "+=",     # 加赋值
    "-=",     # 减赋值
    "*=",     # 乘赋值
    "/=",     # 除赋值
    "%=",     # 取余赋值
    "&=",     # 位与赋值
    "|=",     # 位或赋值
    "^=",     # 位异或赋值
    "<<=",    # 左移赋值
    ">>="     # 右移赋值
]

logical_operators = [
    "==",  # 相等运算符
    "!=",  # 不等运算符
    ">=",  # 大于等于运算符
    "<="   # 小于等于运算符
]


def contains_opt(code):
    for op in operators:
        if op in code:
            if not any(log_op in code for log_op in logical_operators):
                return op 
    return None 


def opt_nodes(code, lab):
    code = code_normal(code)
    opt = contains_opt(code)
    if opt is not None:
        # code = code.replace(' ', '')
        if '--' in code or '++' in code:
            p1 = r'{}[ ]*{}'.format(re.escape(lab), re.escape(opt))
            p2 = r'{}[ ]*{}'.format(re.escape(opt), re.escape(lab))
            r1 = re.search(p1, code)
            r2 = re.search(p2, code)
            if r1 or r2:
                return True
        else:
            p1 = r'{}[ ]*{}'.format(re.escape(lab), re.escape(opt))
            r1 = re.search(p1, code)
            if r1:
                return True

type_nodes = {
    'METHOD': 1,
    'BLOCK':2,
    'PARAM':3,
    'operator':4,
    'operators':15,
    'IDENTIFIER':5,
    'LOCAL':6,
    'LITERAL':7,
    'CONTROL_STRUCTURE':8,
    'JUMP_TARGET':9,
    'RETURN':10,
    'METHOD_RETURN':11,
    'UNKNOWN':12,
    'ELSE':13,
    'DBUG_RETURN':14
}

def read_file(file):
    with open(file, encoding='utf-8') as f:
        content = f.readlines()
        f.close()
    return content

def read_json(file):
    with open(file, 'r') as f:
        files = json.load(f)
        f.close()
    return files

def write_file(file, s, root='newcpgg'):
    os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, os.path.basename(file))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(s))
        f.close()
    return None

def create_argparse(description=None):
    parser = argparse.ArgumentParser(description)
    if description is not None:
        print(description)
    return parser

def load_model(path):
    model = Word2Vec.load(path)
    return model.wv

# def contains_keyword(sentence):
#     sentence = sentence
#     for keyword in type_nodes.keys():
#         if keyword in sentence:
#             return True
#     return False

def contains_keyword(sentence, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return bool(re.search(pattern, sentence))

def extract_code(sentence, keyword):
    code = ''
    sentence = sentence.replace('\\n', '')
    if type_nodes[keyword] == 1:
        code = sentence.split(',')[1]
    elif type_nodes[keyword] == 2:
        code = 'empty'
    elif type_nodes[keyword] == 3:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
    elif type_nodes[keyword] == 4 or type_nodes[keyword] == 15:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
    elif type_nodes[keyword] == 5:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
        code = code[:code.index(',')].split('\\n')[0]
    elif type_nodes[keyword] == 6:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
        # code = code.split(':')[0]
    elif type_nodes[keyword] == 7:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
        code = code[:code.index(',')].split('\\n')[0]
    elif type_nodes[keyword] == 8:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
        code = code[:code.index(',')].split('\\n')[0]
    elif type_nodes[keyword] == 9:
         code = sentence[sentence.index(',')+1:].split('\\n')[0]
    elif type_nodes[keyword] == 10:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
        code = code.split(';')[0]
    elif type_nodes[keyword] == 11 or type_nodes[keyword]==14:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
    elif type_nodes[keyword] == 12:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
        code = remove_duplicates(code)
    elif type_nodes[keyword] == 13:
        code = sentence[sentence.index(',')+1:].split('\\n')[0]
    else:
        print('find')
    return code

def local_keyword(sentence):
    code = sentence[sentence.index(',')+1:].split('\\n')[0]
    c1, c2 = code.split(':')[0], code.split(':')[1]
    c1, c2 = c1.strip(), c2.strip()
    var = c1.split(c2)[-1].strip()
    type_ = c2
    var_ = var.split(' ')[-1]
    type_ = type_ + var.replace(var_, '') 
    return var, type_, var_

def deduplication(sentence):
    tokens = my_tokenizer(sentence)
    first = tokens[0]
    for idx, token in enumerate(tokens):
        if idx == 0:
            continue
        else:
            if token == first:
                token = tokens[:idx]
    return first


def remove_duplicates(input_string):
    # tokens = my_tokenizer(input_string, nolower=True)
    # tokens = input_string.split(' ')[0]
    try:
        tokens = my_tokenizer(input_string, nolower=True)[0]
    except:
        tokens = ''
    # for idx, token in enumerate(tokens):
    #     if ',' in token:
    #         tokens[idx] = token.replace(',', '')
    # start_c = tokens[0]
    # com_c = []
    # for idx, c in enumerate(tokens):
    #     if idx == 0:
    #         continue
    #     if c == start_c:
    #         if tokens[:idx] == tokens[idx:]:
    #             com_c = tokens[:idx]
    #             break
    # if len(com_c) > 0:
    #     end = com_c[-1]
    #     idx = input_string.index(end)
    #     input_string = input_string[:idx+len(end)] 
    # frist = tokens[0]
    # for idx, c in enumerate(tokens):
    #     if idx == 0:
    #         continue
    #     else:
    #         if ',' in c and c.replace(',', '') == frist:
    #             input_string = input_string[:input_string.index(c)]
    #             break
    try:
        input_string = input_string[:input_string.index(',' + tokens)]
        return input_string
    except:
        return input_string

def keyword_code(sentence):
    if ',,' in sentence:
        sentence.replace(',,', ',')
    sentence = sentence
    code = ''
    current_k = 'ELSE'
    for keyword in type_nodes.keys():
        # if keyword in sentence:
        #     current_k = keyword
        #     break
        if contains_keyword(sentence, keyword) and sentence.index(keyword) == 0:
            current_k = keyword
            break
    code = extract_code(sentence, current_k)
    return code

# model_w2v = load_model(model_path)
def word2vec(sentence):
    sentence = sentence.replace('&lt;', '<')
    sentence = sentence.replace('&gt;', '>')
    sentence = sentence.replace('&amp;', '&')
    sentence = sentence.replace('&quot;', '"')
    tokens = my_tokenizer(sentence)
    nrp = np.zeros(100)
    for token in tokens:
        try:
            embedding = model_w2v[token]
        except:
            embedding = np.zeros(100)
        nrp = np.add(nrp, embedding)
    if len(tokens) > 0:
        fnrp = np.divide(nrp, len(tokens))
    else:
        fnrp = nrp
    return fnrp.tolist()

def pre_word2vec(sentence):
    if not isinstance(sentence, list):
        sentence = sentence.replace('&lt;', '<')
        sentence = sentence.replace('&gt;', '>')
        sentence = sentence.replace('&amp;', '&')
        sentence = sentence.replace('&quot;', '"')
    tokens = codebert_tokenizer(sentence, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    # input_ids, attention_mask = encode_input(tokens)
    # with torch.no_grad():
        # outputs = code_model(input_ids.to(device), attention_mask.to(device))
    # last_hidden_states = outputs.last_hidden_state
    # sentence_embedding = last_hidden_states[:, 0, :]
    # return outputs.tolist()
    return tokens
    # return sentence_embedding.tolist()
    
def ecycle(cycle, sn, dn):
    for c in cycle:
        if sn in cycle and dn in cycle:
            return True
    return False

def ncycle(cycle, n):
    for c in cycle:
        if n in cycle:
            return True
    return False

def parse_line(line):
    before, after = line[0], line[1]
    return int(before), int(after)

def parse_line2(line):
    return int(line[0][0])

def code_normal(sentence):
    sentence = sentence.replace('&lt;', '<')
    sentence = sentence.replace('&gt;', '>')
    sentence = sentence.replace('&amp;', '&')
    sentence = sentence.replace('&quot;', '"')
    return sentence

def split_list(lst):
    total_length = len(lst)
    
    part1_length = total_length * 8 // 10
    part2_length = total_length // 10 
    part3_length = total_length // 10  

    part1 = lst[:part1_length]
    part2 = lst[part1_length:part1_length + part2_length]
    part3 = lst[part1_length + part2_length:]

    return part1, part2, part3

if __name__ == '__main__':
    model = Word2Vec.load('/root/autodl-tmp/word2vec/reveal_train_subtoken_data')