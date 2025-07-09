from datetime import datetime
import time
import argparse
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings
import random
warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------- 参数设置 --------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55)
parser.add_argument('--n_atom_hid', type=int, default=128)
parser.add_argument('--rel_total', type=int, default=86)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_epochs', type=int, default=40)
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

# ------------------------ 训练流程 ------------------------ #
def train(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    max_acc = 0
    print('Training starts at', datetime.today())
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            p_score, n_score, probas_pred, closs, ground_truth = do_compute(batch, device, model)
            all_preds.append(probas_pred)
            all_labels.append(ground_truth)

            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            total_loss = loss + 0.05 * closs

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * len(p_score)

        train_loss /= len(train_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_metrics = do_compute_metrics(all_preds, all_labels)

        # 验证集评估
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                p_score, n_score, probas_pred, _, ground_truth = do_compute(batch, device, model)
                val_preds.append(probas_pred)
                val_labels.append(ground_truth)
                loss, _, _ = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_metrics = do_compute_metrics(val_preds, val_labels)

        if val_metrics[0] > max_acc:
            max_acc = val_metrics[0]
            torch.save(model, args.pkl_name)

        if scheduler:
            scheduler.step()

        print(f"[Epoch {epoch}/{n_epochs}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_metrics[0]:.4f}, Val Acc: {val_metrics[0]:.4f}, "
              f"Train AUC: {train_metrics[1]:.4f}, Val AUC: {val_metrics[1]:.4f}")

# ------------------------ 测试流程 ------------------------ #
def test(test_loader, model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            p_score, n_score, probas_pred, _, ground_truth = do_compute(batch, device, model)
            preds.append(probas_pred)
            labels.append(ground_truth)

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc, auroc, f1, precision, recall, int_ap, ap = do_compute_metrics(preds, labels)

    print('\n========== Test Results ==========')
    print(f"Accuracy: {acc:.4f}, AUC: {auroc:.4f}, F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AP: {ap:.4f}, Int-AP: {int_ap:.4f}")

# ------------------------ 主程序入口 ------------------------ #
if __name__ == '__main__':

    set_random_seed(42)

    model = models.DCML_DDI(
        args.n_atom_feats, args.n_atom_hid, args.kge_dim, args.rel_total,
        heads_out_feat_params=[64, 64, 64, 64],
        blocks_params=[2, 2, 2, 2],
        projection_dim=128 
    ).to(device)


    loss_fn = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.96 ** epoch)

    train(model, train_loader, val_loader, loss_fn, optimizer, args.n_epochs, device, scheduler)
    best_model = torch.load(args.pkl_name)
    test(test_loader, best_model)
