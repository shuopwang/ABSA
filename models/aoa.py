from layers.dynamic_rnn import DynamicLSTM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer_v3 import Transformer
import numpy as np


class LocationEncoding(nn.Module):
    def __init__(self, opt):
        super(LocationEncoding, self).__init__()
        self.opt = opt

    def forward(self, x, pos_inx):
        batch_size, seq_len = x.size()[0], x.size()[1]
        #print(x.size())
        weight = self.weight_matrix(pos_inx, batch_size, seq_len).to(self.opt.device)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][0]):
                relative_pos = pos_inx[i][0] - j
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
            for j in range(pos_inx[i][0], pos_inx[i][1] + 1):
                weight[i].append(0)
            for j in range(pos_inx[i][1] + 1, seq_len):
                relative_pos = j - pos_inx[i][1]
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
            #print(len(weight[i]))
        #print(len(weight))
        weight = torch.tensor(weight)
        return weight


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


class AOA(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        self.location = LocationEncoding(opt)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        #
        # self.ctx_transformer = Transformer(len(embedding_matrix), embedding_matrix)
        # self.asp_transformer = Transformer(len(embedding_matrix), embedding_matrix)
        # self.dense = nn.Linear(len(embedding_matrix[0]), opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] # batch_size x seq_len

        # text_mask = text_raw_indices.ne(0).type(torch.float).unsqueeze(-1).byte()
        # aspect_mask = aspect_indices.ne(0).type(torch.float).unsqueeze(-1).byte()

        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_raw_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim

        consider_position = False
        if consider_position:
            aspect_position_text = inputs[2]
            ctx_out = self.location(ctx_out, aspect_position_text)  # batch_size x (ctx)seq_len x 2*hidden_dim

        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp) seq_len x 2*hidden_dim
        # ctx_output = self.ctx_transformer(text_raw_indices, text_mask)
        # ctx_out = ctx_output
        #
        # asp_output = self.asp_transformer(aspect_indices, aspect_mask)
        # asp_out = asp_output
        """
        in this part is for jadore version transformer
        """
        # batch_ctx_pos = np.array([
        #     [pos_i + 1 if w_i != 0 else 0
        #      for pos_i, w_i in enumerate(inst)] for inst in text_raw_indices])
        # batch_ctx_pos = torch.LongTensor(batch_ctx_pos)
        #
        # batch_asp_pos = np.array([
        #     [pos_i + 1 if w_i != 0 else 0
        #      for pos_i, w_i in enumerate(inst)] for inst in aspect_indices])
        # batch_asp_pos = torch.LongTensor(batch_asp_pos)
        #
        # ctx_out = self.ctx_transformer(text_raw_indices, batch_ctx_pos.to(self.opt.device))
        # asp_out = self.asp_transformer(aspect_indices, batch_asp_pos.to(self.opt.device))
        #
        # ctx_out = ctx_out[0]
        # asp_out = asp_out[0]

        # tmp_ctx = []
        # for ctx in ctx_out:
        #     tmp_ctx.append(ctx[:torch.max(ctx_len).item()])
        # ctx_out = torch.stack(tmp_ctx)
        #
        # tmp_asp = []
        # for asp in asp_out:
        #     tmp_asp.append(asp[:torch.max(asp_len).item()])
        # asp_out = torch.stack(tmp_asp)
        """
        end of jadore
        """

        # print(ctx_out.size())
        # print(asp_out.size())
        # asdf
        # print(ctx_out.size())
        # print(asp_out.size())

        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        # print(weighted_sum.size())
        out = self.dense(weighted_sum) # batch_size x polarity_dim

        return out