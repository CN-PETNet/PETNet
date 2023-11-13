from param import *
from data_loader import get_data_loader
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
from model import ModelTransToken
import numpy as np
import random
import datetime


def fix_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    time_begin = datetime.datetime.now()
    fix_seeds(s)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='interval between validation')
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss().to(device)

    dim_seq, dim_meta, dim_token, \
    train_dl, val_dl, test_dl, \
    test_abu_dl, test_drb_dl, test_jm_dl, test_wb_dl = get_data_loader(device, seq_len_max=seq_len_limit, mode=mode_str,
                                                                       use_seq_dim=use_dim_seq,
                                                                       use_meta_dim=use_dim_meta,
                                                                       use_app_dim=use_dim_app)

    model = ModelTransToken(model_name, seq_dim=len(use_dim_seq), app_dim=num_app, meta_dim=dim_meta,
                            hidden_dim=hidden_dim_trans_all, nb_heads=n_heads, nb_layers=n_layers,
                            dropout=0.1, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Training Phase
    if flag_train:
        print("\nStart Training:")
        tot_iters = opt.n_epochs * len(train_dl)
        iter = 0
        acc = [0.0]
        losses = [float('inf')]
        acc_max = -1.0
        recall_max = -1.0
        f1_max = -1.0
        flag_stop = False
        for epoch in range(opt.n_epochs):
            train_loss = 0.0
            if flag_stop:
                break
            for _, (X, Y) in enumerate(train_dl):
                model.train()
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, Y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(Y)
                iter += 1
                if iter % opt.sample_interval == 0:
                    model.eval()
                    pred_all = []
                    gt_all = []
                    val_loss = 0.0
                    for _, (X, Y) in enumerate(val_dl):
                        gt_all.append(Y.cpu().numpy())
                        pred = model(X)
                        val_loss += criterion(pred, Y).item()
                        pred_all.append(pred.detach().cpu().numpy() > 0.5)
                    pred_all = np.concatenate(pred_all, axis=0)
                    gt_all = np.concatenate(gt_all, axis=0)
                    acc = accuracy_score(gt_all, pred_all)
                    recall = recall_score(gt_all, pred_all)
                    f1 = f1_score(gt_all, pred_all)
                    prec = precision_score(gt_all, pred_all)
                    mtx = confusion_matrix(gt_all, pred_all)
                    if acc > acc_max:
                        print('[{}/{}] VAL-Loss:{}'
                              '\tPrecision:{:.4f}\tRecall:{:.4f}'
                              '\tF1:{:.4f}\tAccuracy:{:.4f}'.format(iter, tot_iters, val_loss, prec, recall, f1, acc))
                        print('*Max Acc So Far*')
                        acc_max = acc
                        torch.save(model.state_dict(), model_save_path)
                    if acc_max == 1.0:
                        flag_stop = True
                        break

    # Evaluation phase.
    model.eval()
    model.load_state_dict(torch.load(model_save_path))
    ds_names = ['Test', 'Test-w', 'Test-j', 'Test-a', 'Test-d']
    val_ds = [test_dl, test_wb_dl, test_jm_dl, test_abu_dl, test_drb_dl]

    acc_list = []
    pc_list = []
    rc_list = []
    f1_list = []
    for i in range(len(val_ds)):
        print("\nEvaluation: {} Set".format(ds_names[i]))
        gt_all_test = []
        pred_all_test = []
        test_loss = 0.0
        for _, (X, Y) in enumerate(val_ds[i]):
            gt_all_test.append(Y.cpu().numpy())
            pred_test = model(X)
            test_loss += criterion(pred_test, Y).item()
            pred_all_test.append(pred_test.detach().cpu().numpy() > 0.5)
        pred_all_test = np.concatenate(pred_all_test, axis=0)
        gt_all_test = np.concatenate(gt_all_test, axis=0)
        acc = accuracy_score(gt_all_test, pred_all_test)
        recall = recall_score(gt_all_test, pred_all_test)
        f1 = f1_score(gt_all_test, pred_all_test)
        prec = precision_score(gt_all_test, pred_all_test)
        mtx = confusion_matrix(gt_all_test, pred_all_test)
        acc_list.append(acc)
        pc_list.append(prec)
        rc_list.append(recall)
        f1_list.append(f1)
        print('Accuracy:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\tF1:{:.4f}' \
              '\nConfusion Matrix:\n{}'.format(acc, prec, recall, f1, mtx))

    time_end = datetime.datetime.now()
    print("\nStart: {}\nEnd: {}".format(time_begin, time_end))
