import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import numpy as np
from layers import bert_base


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
        #pos_inx = pos_inx.cpu().numpy()
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
        print(len(weight))
        weight = torch.tensor(weight)
        return weight


class Bert_BaseLine(nn.Module):
    def __init__(self, opt):
        super(Bert_BaseLine, self).__init__()
        self.opt = opt
        self.bert_embedding = bert_base.Bert_Base(opt)
        self.location = LocationEncoding(opt)
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        ctx, asp, asp_len, ctx_len = self.bert_embedding(inputs)
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim
        # ctx_out = ctx
        # asp_out = asp
        # print(ctx_out.size())
        consider_position = True
        if consider_position:
            aspect_position_text = inputs[2]
            aspect_in_text = []
            for aspect_position in aspect_position_text:
                left, right = aspect_position.split('_')
                aspect_in_text.append(np.asarray([int(left), int(right)]))
            aspect_in_text = np.asarray(aspect_in_text)
            ctx_out = self.location(ctx_out, aspect_in_text)  # batch_size x (ctx)seq_len x 2*hidden_dim

        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp) seq_len x 2*hidden_dim

        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum) # batch_size x polarity_dim

        return out

