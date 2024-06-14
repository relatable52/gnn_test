import dgl
import torch
import numpy as np
import os
import random
import pandas 
import bidict
from argparse import ArgumentParser
from utils import load_config

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class Dataset:
    def __init__(self, config, mode):
        save_path = config['save_path']
        graph = dgl.load_graphs(os.path.join(save_path, f'{mode}.pkl'))[0][0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['label'] = graph.ndata['label'].long()
        self.graph = graph
        self.graph = dgl.to_simple(self.graph)

    def split(self, samples=20):
        labels = self.graph.ndata['label']
        n = self.graph.num_nodes()
        if 'mark' in self.graph.ndata:
            index = self.graph.ndata['mark'].nonzero()[:,0].numpy().tolist()
        else:
            index = list(range(n))
        train_masks = torch.zeros([n,20]).bool()
        val_masks = torch.zeros([n,20]).bool()
        test_masks = torch.zeros([n,20]).bool()
        
        train_masks[:,:10] = self.graph.ndata['train_mask'].repeat(10,1).T
        val_masks[:,:10] = self.graph.ndata['val_mask'].repeat(10,1).T
        test_masks[:,:10] = self.graph.ndata['test_mask'].repeat(10,1).T
       
        for i in range(10):
            pos_index = np.where(labels == 1)[0]
            neg_index = list(set(index) - set(pos_index))
            pos_train_idx = np.random.choice(pos_index, size=2*samples, replace=False)
            neg_train_idx = np.random.choice(neg_index, size=8*samples, replace=False)
            train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4*samples]])
            train_masks[train_idx, 10+i] = 1
            val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4*samples:]])
            val_masks[val_idx, 10+i] = 1
            test_masks[index, 10+i] = 1
            test_masks[train_idx, 10+i] = 0
            test_masks[val_idx, 10+i] = 0

        self.graph.ndata['train_masks'] = train_masks
        self.graph.ndata['val_masks'] = val_masks
        self.graph.ndata['test_masks'] = test_masks

def preprocess(config, mode='AF'):
    classes_path = config['classes_path']
    features_path = config['features_path']
    edges_path = config['edges_path']
    save_path = config['save_path']

    labels = pandas.read_csv(classes_path).to_numpy()
    node_features = pandas.read_csv(features_path, header=None).to_numpy()

    node_dict = bidict.bidict()

    for i in range(labels.shape[0]):
        node_dict[i] = labels[i][0]

    new_labels = np.zeros(labels.shape[0]).astype(int)
    marks = labels[:,1]!='unknown'
    features = None
    if mode == 'AL':
        features = node_features[:,1:]
    elif mode == 'LF':
        features = node_features[:,1:95]
    else:
        raise NotImplementedError
    new_labels[labels[:,1]=='1']=1

    train_mask = (features[:,0]<=25)&marks
    val_mask = (features[:,0]>25)&(features[:,0]<=34)&marks
    test_mask = (features[:,0]>34)&marks
    print(train_mask.sum(), val_mask.sum(), test_mask.sum())
    edges = pandas.read_csv(edges_path).to_numpy()

    new_edges = np.zeros_like(edges)

    for i in range(edges.shape[0]):
        new_edges[i][0] = node_dict.inv[edges[i][0]]
        new_edges[i][1] = node_dict.inv[edges[i][1]]

    graph = dgl.graph((new_edges[:,0], new_edges[:,1]))
    graph.ndata['train_mask'] = torch.tensor(train_mask).bool()
    graph.ndata['val_mask'] = torch.tensor(val_mask).bool()
    graph.ndata['test_mask'] = torch.tensor(test_mask).bool()
    graph.ndata['mark'] = torch.tensor(marks).bool()
    graph.ndata['label'] = torch.tensor(new_labels)
    graph.ndata['feature'] = torch.tensor(features)

    dgl.save_graphs(os.path.join(save_path, f'{mode}.pkl'), [graph])

    data = Dataset(config, mode)
    data.split()
    print(data.graph)
    print(data.graph.ndata['train_masks'].sum(0), data.graph.ndata['val_masks'].sum(0), data.graph.ndata['test_masks'].sum(0))
    dgl.save_graphs(os.path.join(save_path, f'{mode}.pkl'), [data.graph])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", type=str, default='AL')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    config = load_config(args.config_path)
    mode = args.mode
    print(f'Processing dataset {mode} ...')
    preprocess(config, mode)
    print('Done')    

if __name__ == "__main__":
    main()
