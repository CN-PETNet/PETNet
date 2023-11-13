import pickle
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from param import *


def get_scaler(file_train, use_seq_dim: list, use_meta_dim: tuple, use_app_dim: list, mode='seq'):
    scaler = StandardScaler()
    train_X = pickle.load(open(file_train, 'rb'))
    train_X_mode = train_X[mode]
    if mode == 'seq':
        train_X_mode_cut = []
        for line in train_X_mode:
            train_X_mode_cut.append(line[:, use_seq_dim])
        train_X_mode = torch.cat(train_X_mode_cut, dim=0)
    elif mode == 'token_int':
        train_X_mode_cut = []
        for line in train_X_mode:
            train_X_mode_cut.append(line[:, use_app_dim])
        train_X_mode = torch.cat(train_X_mode_cut, dim=0)
    elif mode == 'meta':
        use_meta_dim_start = use_meta_dim[0]
        use_meta_dim_end = use_meta_dim[1]
        train_X_mode_cut = []
        for line in train_X_mode:
            train_X_mode_cut.append(line[use_meta_dim_start:use_meta_dim_end])
        train_X_mode = torch.vstack(train_X_mode_cut)
    scaler.fit(train_X_mode)
    return scaler


class Collate:
    def __init__(self, device):
        self.device = device

    def _collate(self, batch):
        xs = [v[0] for v in batch]
        xm = torch.vstack([v[1] for v in batch]).to(self.device)
        xt = [v[2] for v in batch]
        ys = torch.tensor([v[3] for v in batch], device=self.device)
        seq_lengths = torch.LongTensor([v for v in map(len, xs)])
        xs = pad_sequence(xs, batch_first=True)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        token_lengths = seq_lengths
        xt = pad_sequence(xt, batch_first=True)
        xs = xs[perm_idx]
        xm = xm[perm_idx]
        xt = xt[perm_idx]
        ys = ys[perm_idx]
        return (xs, seq_lengths, xm, xt, token_lengths), ys

    def __call__(self, batch):
        return self._collate(batch)


class NetFlowDataset(data.Dataset):
    def __init__(self, mode, ds, scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim: list, use_meta_dim: tuple, use_app_dim: list):
        file_pkl_x = "./Data/ModelInput/{}_x_{}.pkl".format(mode, ds)
        file_pkl_y = "./Data/ModelInput/{}_y_{}.pkl".format(mode, ds)
        X_raw = pickle.load(open(file_pkl_x, 'rb'))
        self.X_seq = []
        self.X_meta = []
        self.X_token = []
        for x_seq in X_raw['seq']:
            x_seq = x_seq[:, use_seq_dim]
            self.X_seq.append(torch.FloatTensor(scaler_seq.transform(x_seq)[:min(seq_len_max, len(x_seq)), :]).to(device))
        for x_meta in X_raw['meta']:
            use_meta_dim_start = use_meta_dim[0]
            use_meta_dim_end = use_meta_dim[1]
            x_meta = x_meta[use_meta_dim_start:use_meta_dim_end]
            self.X_meta.append(torch.FloatTensor(scaler_meta.transform(x_meta.reshape(1, -1))).to(device))
        for x_token in X_raw['token_int']:
            x_token = x_token[:, use_app_dim]
            self.X_token.append(x_token.int()[:min(seq_len_max, len(x_token)), :].to(device))
        self.dim_seq = self.X_seq[-1].shape[1]
        self.dim_meta = self.X_meta[-1].shape[1]
        self.dim_token = self.X_token[-1].shape[1]
        self.Y = torch.FloatTensor(pickle.load(open(file_pkl_y, 'rb'))).to(device)

    def get_dim(self):
        return self.dim_seq, self.dim_meta, self.dim_token

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_meta[idx], self.X_token[idx], self.Y[idx]


def get_data_loader(device, seq_len_max, mode, use_seq_dim: list, use_meta_dim: tuple, use_app_dim: list):
    scaler_train = "./Data/ModelInput/%s_x_train.pkl" % mode
    scaler_seq = get_scaler(scaler_train, use_seq_dim, use_meta_dim, use_app_dim, mode='seq')
    scaler_meta = get_scaler(scaler_train, use_seq_dim, use_meta_dim, use_app_dim, mode='meta')
    scaler_token = None
    print("\nLoad Data: NetFlowDataset")
    train_ds = NetFlowDataset(mode, 'train', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    val_ds = NetFlowDataset(mode, 'valid', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    test_ds = NetFlowDataset(mode, 'test', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    val_abu_ds = NetFlowDataset(mode, 'test_a', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    val_drb_ds = NetFlowDataset(mode, 'test_d', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    val_jm_ds = NetFlowDataset(mode, 'test_j', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    val_wb_ds = NetFlowDataset(mode, 'test_w', scaler_seq, scaler_meta, scaler_token, device, seq_len_max, use_seq_dim, use_meta_dim, use_app_dim)
    print("Load Data: data.DataLoader - Collate")
    train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collate(device))
    val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(device))
    test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(device))
    val_abu_dl = data.DataLoader(val_abu_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(device))
    val_drb_dl = data.DataLoader(val_drb_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(device))
    val_jm_dl = data.DataLoader(val_jm_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(device))
    val_wb_dl = data.DataLoader(val_wb_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collate(device))
    dim_seq, dim_meta, dim_token = train_ds.get_dim()
    return dim_seq, dim_meta, dim_token, train_dl, val_dl, test_dl, val_abu_dl, val_drb_dl, val_jm_dl, val_wb_dl
