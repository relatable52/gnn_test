from models.gnn import *
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)
import dgl
import numpy as np
import pandas as pd
import itertools
import psutil
import os
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models

class BaseDetector(object):
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        graph = self.data.graph.to(self.train_config['device'])
        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].bool()
        self.val_mask = graph.ndata['val_mask'].bool()
        self.test_mask = graph.ndata['test_mask'].bool()
        self.weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()
        self.source_graph = graph
        print("Inductive training:", train_config['inductive'])
        if train_config['inductive'] == False:
            self.train_graph = graph
            self.val_graph = graph
        else:
            self.train_graph = graph.subgraph(self.train_mask)
            self.val_graph = graph.subgraph(self.train_mask+self.val_mask)
        self.best_score = -1
        self.patience_knt = 0

    def train(self):
        pass

    def eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            score['roc_curve'] = roc_curve(labels, probs)
            score['pr_curve'] = precision_recall_curve(labels, probs)
            preds = np.rint(probs)
            score['classification_report'] = classification_report(labels, preds, output_dict=True)
            labels = np.array(labels)
            k = labels.sum()
        score['RecK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
        return score
    
class BaseGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph)
            loss = F.cross_entropy(logits[self.train_graph.ndata['train_mask']], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score