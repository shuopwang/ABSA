import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import jieba
import fasttext


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def build_tokenizer(fnames, max_seq_len, dat_fname, opt):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        tokenizer = Tokenizer(max_seq_len)
        if opt.chinese:
            vocab = ['<pad>', '<unk>'] + list(opt.fasttext_word_embedding_model.words)
            tokenizer.idx = 1
            print('The size of total vocab size is: {}'.format(len(vocab)))
            tokenizer.word2idx = {word: token for token, word in enumerate(vocab)}
            tokenizer.idx2word = {token: word for word, token in tokenizer.word2idx.items()}
            pickle.dump(tokenizer, open(dat_fname, 'wb'))
            pickle.dump(tokenizer.word2idx, open('word2idx.dat', 'wb'))
            return tokenizer
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
            tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, opt):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        print(len(word2idx))
        if opt.chinese:
            embedding_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim,))
            for word, i in word2idx.items():
                vec = opt.fasttext_word_embedding_model[word]
                if vec is not None:
                    embedding_matrix[i] = opt.fasttext_word_embedding_model[word]

        else:
            fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
                if embed_dim != 300 else './glove.42B.300d.txt'

            word_vec = _load_word_vec(fname, word2idx=word2idx)
            print('building embedding_matrix:', dat_fname)
            for word, i in word2idx.items():
                vec = word_vec.get(word)
                if vec is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
        sdf
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', chinese=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        if chinese:
            unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len=None, use_bert=False, use_chinese=False):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        if use_bert:
            unique_id = 0
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()

                text_raw_indices = self.convert_examples_to_features(unique_id, text_left + " " + aspect + " " + text_right, tokenizer, max_seq_len)
                text_raw_without_aspect_indices = self.convert_examples_to_features(unique_id, text_left + " " + text_right, tokenizer, max_seq_len)
                text_left_indices = self.convert_examples_to_features(unique_id, text_left, tokenizer, max_seq_len)
                text_left_with_aspect_indices = self.convert_examples_to_features(unique_id, text_left + " " + aspect, tokenizer, max_seq_len)
                text_right_indices = self.convert_examples_to_features(unique_id, text_right, tokenizer, max_seq_len, reverse=True)
                text_right_with_aspect_indices = self.convert_examples_to_features(unique_id, " " + aspect + " " + text_right, tokenizer, max_seq_len, reverse=True)
                #aspect_indices = self.convert_examples_to_features(unique_id, aspect, tokenizer, max_seq_len)

                left_context_len = np.count_nonzero(text_left_indices.input_mask)
                left_context_with_aspect_len = np.count_nonzero(text_left_with_aspect_indices.input_mask)
                #aspect_len = np.sum(aspect_indices.input_mask != 0)
                aspect_in_text = str(left_context_len - 1)+'_'+str(left_context_with_aspect_len - 1)

                # print(text_raw_indices.tokens)
                # print(text_raw_indices.tokens[left_context_len - 1: left_context_with_aspect_len - 1])
                # print(aspect)
                # asdf
                polarity = int(polarity) + 1
                unique_id += 1
                data = {
                    'text_raw_indices': text_left + " " + aspect + " " + text_right,
                    'text_raw_without_aspect_indices': text_left + " " + text_right,
                    'text_left_indices': text_left,
                    'text_left_with_aspect_indices': text_left + " " + aspect,
                    'text_right_indices': text_right,
                    'text_right_with_aspect_indices': " " + aspect + " " + text_right,

                    'text_raw_indices_mask': torch.tensor(text_raw_indices.input_mask, dtype=torch.long),
                    'text_raw_without_aspect_indices_mask': torch.tensor(text_raw_without_aspect_indices.input_mask, dtype=torch.long),
                    'text_left_indices_mask': torch.tensor(text_left_indices.input_mask, dtype=torch.long),
                    'text_left_with_aspect_indices_mask':torch.tensor(text_left_with_aspect_indices.input_mask, dtype=torch.long),
                    'text_right_indices_mask': torch.tensor(text_right_indices.input_mask, dtype=torch.long),
                    'text_right_with_aspect_indices_mask': torch.tensor(text_right_with_aspect_indices.input_mask, dtype=torch.long),
                    'aspect_in_text': aspect_in_text,
                    'polarity': polarity,
                }

                all_data.append(data)

        else:
            if use_chinese:
                for i in range(len(lines)):
                    line = lines[i]
                    # try:
                    line = line.split('\t')
                    point = -1
                    polarity = line[point]
                    polarity = polarity.replace('\n', '')

                    content = line[:point]
                    content = ' '.join(content)
                    content = content.strip()
                    try:
                        polarity = int(polarity)
                    except:
                        continue
                        # if int(polarity) < 0:
                    #         print(line)
                    # except:
                    #     print(i)
                    #     continue
                    try:
                        aspect = re.search('【.*?】', content).group(0)
                    except:
                        continue
                    text_left, _, text_right = [s.lower().strip() for s in content.partition(aspect)]

                    text_left_list = jieba.lcut(text_left)
                    text_right_list = jieba.lcut(text_right)
                    text_right = ' '.join(t for t in text_right_list if t != ' ')
                    text_left = ' '.join(t for t in text_left_list if t != ' ')

                    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right, chinese=use_chinese)
                    text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right, chinese=use_chinese)
                    text_left_indices = tokenizer.text_to_sequence(text_left, chinese=use_chinese)
                    text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect, chinese=use_chinese)
                    text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True, chinese=use_chinese)
                    text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True, chinese=use_chinese)
                    aspect_indices = tokenizer.text_to_sequence(aspect, chinese=use_chinese)
                    left_context_len = np.sum(text_left_indices != 0)
                    aspect_len = np.sum(aspect_indices != 0)
                    aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
                    if polarity == 3:
                        polarity = 0
                    data = {
                        'text_raw_indices': (text_raw_indices),
                        'text_raw_without_aspect_indices': (text_raw_without_aspect_indices),
                        'text_left_indices': (text_left_indices),
                        'text_left_with_aspect_indices': (text_left_with_aspect_indices),
                        'text_right_indices': (text_right_indices),
                        'text_right_with_aspect_indices': (text_right_with_aspect_indices),
                        'aspect_indices': (aspect_indices),
                        'aspect_in_text': aspect_in_text,
                        'polarity': polarity,
                    }

                    all_data.append(data)
            else:
                for i in range(0, len(lines), 3):
                    text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                    aspect = lines[i + 1].lower().strip()
                    polarity = lines[i + 2].strip()

                    if use_chinese:
                        text_left_list = jieba.lcut(text_left)
                        text_right_list = jieba.lcut(text_right)
                        text_right = ' '.join(text_right_list)
                        text_left = ' '.join(text_left_list)

                    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right, chinese=use_chinese)
                    text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right, chinese=use_chinese)
                    text_left_indices = tokenizer.text_to_sequence(text_left, chinese=use_chinese)
                    text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect, chinese=use_chinese)
                    text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True, chinese=use_chinese)
                    text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True, chinese=use_chinese)
                    aspect_indices = tokenizer.text_to_sequence(aspect, chinese=use_chinese)
                    left_context_len = np.sum(text_left_indices != 0)
                    aspect_len = np.sum(aspect_indices != 0)
                    aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
                    polarity = int(polarity) + 1

                    data = {
                        'text_raw_indices': (text_raw_indices),
                        'text_raw_without_aspect_indices': (text_raw_without_aspect_indices),
                        'text_left_indices': (text_left_indices),
                        'text_left_with_aspect_indices': (text_left_with_aspect_indices),
                        'text_right_indices': (text_right_indices),
                        'text_right_with_aspect_indices': (text_right_with_aspect_indices),
                        'aspect_indices': (aspect_indices),
                        'aspect_in_text': aspect_in_text,
                        'polarity': polarity,
                    }

                    all_data.append(data)
        self.data = all_data

    def read_examples(self, line, unique_id):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        #unique_id = 0
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        #unique_id += 1
        return examples

    def convert_examples_to_features(self, unique_id, text, tokenizer, seq_length, reverse=False):
        if reverse:
            text = text[::-1]
        # print(text)
        tokens_a = tokenizer.tokenize(text)
        # print(tokens_a)
        # print(seq_length)
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return InputFeatures(
                unique_id=unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
