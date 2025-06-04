import json
import torch
import dgl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
from glob import glob 

pad_thing_100 = [0.0] * 100
pad_thing_768 = [0.0] * 768
pad_thing_200 = [0.0] * 200

class Custome_dataseet(Dataset):
    def __init__(self, path) -> None:
        super(Custome_dataseet, self).__init__()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        g = dgl.graph(([], []), num_nodes=500)
        data = self.data[index]
        features = data['node_features']
        etype = data['etype']
        edges  = data['graph']
        for i in range(500 - len(features)):
            features.append(pad_thing)
        features = torch.tensor(features, dtype=torch.float32)         
        g.add_edges(edges[0], edges[1])
        target = torch.tensor(data['target'], dtype=torch.float32)
        etype = torch.tensor(etype, dtype=torch.int)
        g.edata['etype'] = etype
        g.ndata['feat'] = features
        return g, target
    
class Gdataset(Dataset):
    def __init__(self, path):
        super(Gdataset, self).__init__()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        g = dgl.graph(([],[]), num_nodes=500)
        data = self.data[index]
        features = data['node_features']
        ntype = data['ntype']
        mask = [[1]] * len(features)
        for i in range(500 - len(features)):
            features.append(pad_thing)
            mask.append([0])
            ntype.append(0)
        features = torch.tensor(features, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        graph = data['graph']
        g.add_edges(graph[0], graph[1])
        target = torch.tensor(data['target'], dtype=torch.float32)
        etype = torch.tensor(data['etype'], dtype=torch.int)
        ntype = torch.tensor(ntype, dtype=torch.int)
        g.edata['type'] = etype
        g.ndata['feat'] = features
        g.ndata['mask'] = mask
        g.ndata['type'] = ntype
        return g, target    

  
class SingleDateset(Dataset):
    def __init__(self, path):
        super(SingleDateset, self).__init__()
        with open(path, 'r') as f:
            self.data = json.load(f)
            f.close()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]['file_path']
        # path = path.replace('train_reveal', 'train_reveal2')
        g = dgl.graph(([], []), num_nodes=500)
        with open(path, 'r') as f:
            data = json.load(f)
            f.close()
        features = data['node_features']
        ntype = data['ntype']
        mask = [[1]] * len(features)
        for i in range(500 - len(features)):
            features.append(pad_thing_100)
            mask.append([0])
            ntype.append(0)
        features = torch.tensor(features, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        graph = data['graph']
        g.add_edges(graph[0], graph[1])
        target = torch.tensor(data['target'], dtype=torch.float32)
        etype = torch.tensor(data['etype'], dtype=torch.int)
        ntype = torch.tensor(ntype, dtype=torch.int)
        g.edata['type'] = etype
        g.ndata['feat'] = features
        g.ndata['mask'] = mask
        g.ndata['type'] = ntype
        # for_ = torch.tensor(data['for'], dtype=torch.int)
        return g, target    

class BalancedDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.pos_samples = [data[i] for i in range(len(data)) if data[i]['target'] == 0]
        self.neg_samples = [data[i] for i in range(len(data)) if data[i]['target'] == 1]
        self.div = len(self.pos_samples) // len(self.neg_samples) 
        self.neg_samples = self.neg_samples * self.div
        self.data = self.neg_samples + self.pos_samples
        random.shuffle(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        g = dgl.graph(([], []), num_nodes=500)
        data = self.data[index]
        features = data['features']
        etype = data['etype']
        edges  = data['edges']
        for i in range(500 - len(features)):
            features.append(pad_thing)
        features = torch.tensor(features, dtype=torch.float32)         
        g.add_edges(edges[0], edges[1])
        target = torch.tensor(data['target'], dtype=torch.float32)
        etype = torch.tensor(etype, dtype=torch.int)
        g.edata['etype'] = etype
        g.ndata['feat'] = features
        return g, target
    
    
  
class JointDataset(Dataset):
    def __init__(self, path):
        super(JointDataset, self).__init__()
        with open(path, 'r') as f:
            self.data = json.load(f)
            f.close()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path1 = self.data[idx]['file_path']
        path2 = path1.replace('train_reveal2', 'reveal_pt')
        path2 = path2.replace('.json', '.pt')
        g = dgl.graph(([], []), num_nodes=500)
        with open(path1, 'r') as f:
            data = json.load(f)
            f.close()
        features = data['node_features']
        ntype = data['ntype']
        mask = [[1]] * len(features)
        for i in range(500 - len(features)):
            features.append(pad_thing_100)
            mask.append([0])
            ntype.append(0)
        features = torch.tensor(features, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        graph = data['graph']
        g.add_edges(graph[0], graph[1])
        target = torch.tensor(data['target'], dtype=torch.float32)
        etype = torch.tensor(data['etype'], dtype=torch.int)
        ntype = torch.tensor(ntype, dtype=torch.int)
        g.edata['type'] = etype
        g.ndata['feat'] = features
        g.ndata['mask'] = mask
        g.ndata['type'] = ntype
        feat = torch.load(path2)
        # label = 0 if 'no-vul' in pt_path else 1
        return g, target, feat
        # return feat, label
    
    
def collate_fn2(batch):
    ginputs, targets, feats = zip(*batch)
    ginputs = dgl.batch(ginputs)
    targets = torch.stack(targets)
    inputs = [item['input_ids'] for item in feats]
    masks = [item['attention_mask'] for item in feats]
    return ginputs, targets, {
        'input_ids': torch.stack(inputs),
        'attention_mask': torch.stack(masks)
    }
    

def collate_fn(batch):
    ginputs, targets = zip(*batch)
    ginputs = dgl.batch(ginputs)
    targets = torch.stack(targets)
    return ginputs, targets


class pt_dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        file_path = self.data[index]['file_path']
        file_path = file_path.replace('train_reveal2', 'reveal_pt')
        file_path = file_path.replace('.json', '.pt')
        feat = torch.load(file_path)
        label = 0 if 'novul' in file_path else 1
        label = torch.tensor(label, dtype=torch.long)
        return feat, label

# test_dataset = pt_dataset('pt_test_file.json')
# train_dataset = pt_dataset('pt_train_file.json')
# valid_dataset = pt_dataset('pt_valid_file.json')

# test_dataset = Gdataset('devign_test_data300.json')
# train_dataset = Gdataset('devign_train_data300.json')
# valid_dataset = Gdataset('devign_valid_data300.json')


# test_dataset = JointDataset('test_reveal_new.json')
# train_dataset = JointDataset('train_reveal_new.json')
# valid_dataset = JointDataset('valid_reveal_new.json')


test_dataset = SingleDateset('test_diverse.json')
train_dataset = SingleDateset('train_diverse.json')
valid_dataset = SingleDateset('valid_diverse.json')

if __name__ == '__main__':
    test_dataset = SingleDateset('test_diverse.json')
    test_loader = DataLoader(dataset=test_dataset, batch_size=32,
                        shuffle=True, num_workers=1, pin_memory=True, collate_fn=collate_fn)
    for input, label in tqdm(test_loader):
        print(label)