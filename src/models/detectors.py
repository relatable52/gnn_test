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
    
class XGBoostDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)], verbose=False)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score
    
class RFDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score

class BGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # gnn = globals()[model_config['model']]
        self.depth = 6 if 'depth' not in model_config else model_config['depth']
        self.iter_per_epoch = 10 if 'iter_per_epoch' not in model_config else model_config['iter_per_epoch']
        self.gbdt_alpha = 1 if 'gbdt_alpha' not in model_config else model_config['gbdt_alpha']
        self.gbdt_lr = 0.1 if 'gbdt_lr' not in model_config else model_config['gbdt_lr']
        self.train_non_gbdt = False if 'train_non_gbdt' not in model_config else model_config['train_non_gbdt']
        self.only_gbdt = False if 'only_gdbt' not in model_config else model_config['only_gdbt']
        self.normalize_features = False if 'nomarlize_features' not in model_config else model_config['normalize_features']

        if not self.only_gbdt:
            model_config['in_feats'] = self.source_graph.ndata['feature'].shape[1] + self.labels.unique().shape[0]
        else:
            model_config['in_feats'] = self.labels.unique().size(0)

        self.model = GCN(**model_config).to(train_config['device'])
        self.gbdt_model = None
    
    def preprocess(self):
        gbdt_X_train = pd.DataFrame(self.source_graph.ndata['feature'][self.train_mask].cpu().numpy())
        gbdt_y_train = pd.DataFrame(self.labels[self.train_mask].cpu().numpy()).astype(float)

        raw_X = pd.DataFrame(self.source_graph.ndata['feature'].clone().cpu().numpy())
        encoded_X = self.source_graph.ndata['feature'].clone()
        if not self.only_gbdt and self.normalize_features:
            min_vals, _ = torch.min(encoded_X[self.train_mask], dim=0, keepdim=True)
            max_vals, _ = torch.max(encoded_X[self.train_mask], dim=0, keepdim=True)
            encoded_X[self.train_mask] = (encoded_X[self.train_mask] - min_vals) / (max_vals - min_vals)
            encoded_X[self.val_mask | self.test_mask] = (encoded_X[self.val_mask | self.test_mask] - min_vals) / (max_vals - min_vals)
            if encoded_X.isnan().any():
                row, col = torch.where(encoded_X.isnan())
                encoded_X[row, col] = self.source_graph.ndata['feature'][row, col]
            if encoded_X.isinf().any():
                row, col = torch.where(encoded_X.isinf())
                encoded_X[row, col] = self.source_graph.ndata['feature'][row, col]

        node_features = torch.empty(encoded_X.shape[0], self.model_config['in_feats'], requires_grad=True, device=self.labels.device)
        if not self.only_gbdt:
            node_features.data[:, :-2] = self.source_graph.ndata['feature'].clone()
        self.source_graph.ndata['feature'] = node_features
        return gbdt_X_train, gbdt_y_train, raw_X, encoded_X

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, epoch):
        pool = Pool(gbdt_X_train, gbdt_y_train)
        if epoch == 0:
            catboost_model_obj = CatBoostClassifier
            catboost_loss_fn = 'MultiClass'
        else:
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'MultiRMSE'
        
        epoch_gbdt_model = catboost_model_obj(iterations=self.iter_per_epoch,
                                              depth=self.depth,
                                              learning_rate=self.gbdt_lr,
                                              loss_function=catboost_loss_fn,
                                              random_seed=0,
                                              nan_mode='Min')
        epoch_gbdt_model.fit(pool, verbose=False)
        
        if epoch == 0:
            self.base_gbdt = epoch_gbdt_model
        else:
            if self.gbdt_model is None:
                self.gbdt_model = epoch_gbdt_model
            else:
                self.gbdt_model = sum_models([self.gbdt_model, epoch_gbdt_model], weights=[1, self.gbdt_alpha])
                # self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, self.gbdt_alpha])

    def update_node_features(self, X, encoded_X):
        predictions = self.base_gbdt.predict_proba(X)
        # predictions = self.base_gbdt.predict(X, prediction_type='RawFormulaVal')
        if self.gbdt_model is not None:
            predictions_after_one = self.gbdt_model.predict(X)
            predictions += predictions_after_one

        predictions = torch.tensor(predictions, device=self.labels.device)
        node_features = self.source_graph.ndata['feature']
        if not self.only_gbdt:
            if self.train_non_gbdt:
                predictions = torch.concat((node_features.detach().data[:, :-2], predictions), dim=1)
            else:
                predictions = torch.concat((encoded_X, predictions), dim=1)
        node_features.data = predictions.float().data

    def train(self):
        gbdt_X_train, gbdt_y_train, raw_X, encoded_X = self.preprocess()
        optimizer = torch.optim.Adam(
            itertools.chain(*[self.model.parameters(), [self.source_graph.ndata['feature']]]), lr=self.model_config['lr']
        )
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.train_gbdt(gbdt_X_train, gbdt_y_train, e)
            self.update_node_features(raw_X, encoded_X)
            node_features_before = self.source_graph.ndata['feature'].clone()
            
            self.model.train()
            for _ in range(self.iter_per_epoch):
                logits = self.model(self.source_graph)
                loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
            
            # Update GBDT target
            gbdt_y_train = (self.source_graph.ndata['feature'] - node_features_before)[self.train_mask, -2:].detach().cpu().numpy()
            
            # Check if update is frozen
            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Nodes do not change anymore. Stopping...')
                break
        return test_score
