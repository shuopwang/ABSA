import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Bert_Base(nn.Module):
    def __init__(self, opt):
        super(Bert_Base, self).__init__()
        self.opt = opt
        #self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def forward(self, inputs, use_hidden_state=False):
        text_raw_indices, text_raw_indices_mask, aspect_position_text = inputs[0], inputs[1], inputs[2]
        ctx = self.opt.bse.get_vector(text_raw_indices)

        ctx_len = torch.sum(text_raw_indices_mask != 0, dim=1)
        vectors = []
        aspect_vectors = []
        asp_len = []
        for idx, vector in enumerate(ctx):
            # print(aspect_position_text[idx])
            # print(vector.size())
            #vector = torch.stack(vector)
            left, right = aspect_position_text[idx].split('_')
            vector = [np.asarray(each, dtype=float) for each in vector]
            aspect_vector = vector[int(left):int(right)]
            # if self.opt.device:
            #     vector = vector.cpu()
            #     aspect_vector = aspect_vector.cpu()

            pad_number = self.opt.max_seq_len - len(vector) + 2
            #ctx_len.append(len(vector))
            vector = np.asarray(vector, dtype=float)
            vector = vector[1:-1]
            vector = np.concatenate((vector, np.zeros((pad_number, self.opt.embed_dim))))
            vector = vector.astype('float32')
            vector = torch.from_numpy(vector)
            #pad_tuple = (0, 0, left, 0)
            #vector = F.pad(vector, pad_tuple, 'constant', 0)

            pad_number = self.opt.max_seq_len - len(aspect_vector)
            asp_len.append(len(aspect_vector))
            aspect_vector = np.asarray(aspect_vector)
            aspect_vector = np.concatenate((aspect_vector, np.zeros((pad_number, self.opt.embed_dim))))
            aspect_vector = aspect_vector.astype('float32')
            aspect_vector = torch.from_numpy(aspect_vector)
            if self.opt.device:
                vector = vector.to(self.opt.device)
                aspect_vector = aspect_vector.to(self.opt.device)
            vectors.append(vector)
            aspect_vectors.append(aspect_vector)
        ctx = torch.stack(vectors)
        asp = torch.stack(aspect_vectors)
        asp_len = torch.from_numpy(np.asarray(asp_len))
        #ctx_len = torch.from_numpy(np.asarray(ctx_len))
        if self.opt.device:
            asp_len = asp_len.to(self.opt.device)
            ctx_len = ctx_len.to(self.opt.device)
        ctx.requires_grad = False
        asp.requires_grad = False
        # print(vectors.size())
        # print(aspect_vectors.size())

        return ctx, asp, ctx_len, asp_len
