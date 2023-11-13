import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ModelTransToken(nn.Module):
    def __init__(self, model_name, seq_dim, app_dim, hidden_dim, meta_dim, nb_heads, nb_layers, dropout, device):
        super(ModelTransToken, self).__init__()
        self.model_name = model_name
        self.seq_dim = seq_dim
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
        self.nb_heads = nb_heads
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = TransformerEncoderLayer(self.hidden_dim, nb_heads, self.hidden_dim, dropout)
        self.enc_token_max_len = app_dim
        dict_size = 65536
        self.token_embed_size = 16
        self.token_embedding = nn.Embedding(dict_size, self.token_embed_size)
        self.enc_token_seq_encoder_nb_heads = 1
        self.enc_token_ffn_dim = 4
        nb_token_seq_layers = 1
        self.enc_token_seq_pos_encoder = PositionalEncoding(self.token_embed_size, dropout)
        enc_token_seq_encoder_layers = TransformerEncoderLayer(self.token_embed_size,
                                                               self.enc_token_seq_encoder_nb_heads,
                                                               self.enc_token_ffn_dim, dropout)
        self.enc_token_seq_transformer_encoder = TransformerEncoder(enc_token_seq_encoder_layers, nb_token_seq_layers)
        if self.model_name.startswith('CSHierAttn'):
            attn_dim = 16
            self.v = nn.Linear(attn_dim, 1, bias=False)
            self.W_meta = nn.Linear(self.meta_dim, attn_dim, bias=False)
            self.W_trans_out = nn.Linear(self.token_embed_size, attn_dim, bias=False)
        self.inp_embedding = nn.Linear(self.seq_dim + self.token_embed_size, self.hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nb_layers)
        self.out_fc = nn.Linear(self.meta_dim + self.hidden_dim, 1)
        self.device = device

    def forward(self, X):
        X_seq, seq_lens, X_meta, enc_seq, _ = X
        nb_batches = len(seq_lens)
        enc_token_lengths = [self.enc_token_max_len] * enc_seq.size(1) * enc_seq.size(0)
        enc_token_key_padding_mask = self.generate_key_padding_mask(enc_token_lengths)
        enc_token_attn_mask = self.generate_attn_mask(self.enc_token_seq_encoder_nb_heads, enc_token_lengths)
        enc_seq = enc_seq.view(-1, self.enc_token_max_len)
        enc_seq_embed = self.token_embedding(enc_seq).transpose(0, 1)
        enc_seq_embed = self.enc_token_seq_pos_encoder(enc_seq_embed)
        enc_seq_trans_out = self.enc_token_seq_transformer_encoder(enc_seq_embed, mask=enc_token_attn_mask,
                                                                   src_key_padding_mask=enc_token_key_padding_mask).transpose(0, 1).view(nb_batches, -1, self.enc_token_max_len, self.token_embed_size)
        if self.model_name.startswith('CSHierAttn'):
            energy = self.v(torch.tanh(torch.unsqueeze(torch.unsqueeze(self.W_meta(X_meta), 1), 1) + self.W_trans_out(enc_seq_trans_out)))
            prob = F.softmax(energy, dim=2)
            enc_token_seq_out = torch.sum(prob * enc_seq_trans_out, dim=2)
        else:
            enc_token_seq_out = enc_seq_trans_out[:, :, 0, :]
        X_seq = torch.cat((X_seq, enc_token_seq_out), 2)
        nb_batches = len(seq_lens)
        key_padding_mask = self.generate_key_padding_mask(seq_lens)
        attn_mask = self.generate_attn_mask(self.nb_heads, seq_lens)
        pkt_seq_embed = self.inp_embedding(X_seq).transpose(0, 1)
        pkt_seq_embed = self.pos_encoder(pkt_seq_embed)
        trans_out = self.transformer_encoder(pkt_seq_embed, mask=attn_mask,
                                             src_key_padding_mask=key_padding_mask).transpose(0, 1)
        out = []
        for i in range(nb_batches):
            out.append(trans_out[i, seq_lens[i] - 1, :])
        out = torch.vstack(out)
        if not self.model_name.startswith('no_meta'):
            out = torch.cat([out, X_meta], dim=1)
        final_out = torch.sigmoid(torch.squeeze(self.out_fc(out), dim=1))
        return final_out

    def generate_key_padding_mask(self, data_length):
        bsz = len(data_length)
        max_len = max(data_length)
        key_padding_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
        for i in range(bsz):
            key_padding_mask[i, data_length[i]:] = True
        return key_padding_mask.to(self.device)

    def generate_attn_mask(self, nb_heads, data_length):
        bsz = len(data_length)
        max_len = max(data_length)
        attn_mask = torch.ones((bsz * nb_heads, max_len, max_len), dtype=torch.bool)
        for i in range(bsz):
            attn_mask[i * nb_heads:(i + 1) * nb_heads, :, :data_length[i]] = False
        return attn_mask.to(self.device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

