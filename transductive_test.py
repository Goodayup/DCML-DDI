from datetime import datetime
import time
import argparse
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from models import MVN_DDI
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings


warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------- 参数设置 --------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55)
parser.add_argument('--n_atom_hid', type=int, default=128)
parser.add_argument('--rel_total', type=int, default=86)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--kge_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='transductive_drugbank.pkl')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)

# ------------------------ 数据集加载 ------------------------ #
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = [(t[0], t[1], int(t[2])) for t in data[train_index]]
    val_tup = [(t[0], t[1], int(t[2])) for t in data[val_index]]
    return train_tup, val_tup

df_train = pd.read_csv('drugbank_test/drugbank/fold0/train.csv')
df_test = pd.read_csv('drugbank_test/drugbank/fold0/test.csv')

train_triples = [(h, t, r) for h, t, r in zip(df_train['d1'], df_train['d2'], df_train['type'])]
train_tup, val_tup = split_train_valid(train_triples, fold=2, val_ratio=0.2)
test_tup = [(h, t, r) for h, t, r in zip(df_test['d1'], df_test['d2'], df_test['type'])]

train_data = DrugDataset(train_tup, ratio=args.data_size_ratio, neg_ent=args.neg_samples)
val_data = DrugDataset(val_tup, ratio=args.data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

train_loader = DrugDataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DrugDataLoader(val_data, batch_size=args.batch_size * 3, num_workers=2)
test_loader = DrugDataLoader(test_data, batch_size=args.batch_size * 3, num_workers=2)

# ------------------------ 核心函数定义 ------------------------ #
def do_compute(batch, device, model):
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [t.to(device) for t in pos_tri]
    p_score, closs = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [t.to(device) for t in neg_tri]
    n_score, _ = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, closs, ground_truth

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1 = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, _ = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)
    return acc, auroc, f1, precision, recall, int_ap, ap
def test(test_data_loader,model):
    model.eval()
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            p_score, n_score, probas_pred,closs, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')


# 加载保存的参数（state_dict）
loaded_obj = torch.load(args.pkl_name)

# 最终测试
test(test_loader, best_model)




