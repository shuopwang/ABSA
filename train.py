from sklearn import metrics
from data_utils import build_tokenizer, build_embedding_matrix, ABSADataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import math
import os
from bert_sent_encoding import bert_sent_encoding
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN
from models.bert_baseline import Bert_BaseLine
from models.bert_mgan import Bert_MGAN
import jieba
import fasttext


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        max_seq_len = None
        if opt.use_bert:
            model_path = os.path.join(os.path.dirname(__file__), './bert_pretrain_model/uncased_L-12_H-768_A-12/')
            opt.bse = bert_sent_encoding(model_path=model_path,
                                         seq_length=opt.max_seq_len, batch_size=opt.batch_size, word_vector=True, layer=-1)

            max_seq_len = opt.max_seq_len
            self.model = opt.model_class(opt).to(opt.device)
            trainset = ABSADataset(opt.dataset_file['train'], opt.bse.tokenizer, max_seq_len=max_seq_len, use_bert=opt.use_bert)
            testset = ABSADataset(opt.dataset_file['test'], opt.bse.tokenizer, max_seq_len=max_seq_len, use_bert=opt.use_bert)
            self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
            self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_new_tokenizer.dat'.format(opt.dataset), opt=self.opt)
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_embedding_matrix.dat'.format(opt.dataset), opt=self.opt)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = ABSADataset(opt.dataset_file['train'], tokenizer, max_seq_len=max_seq_len, use_bert=opt.use_bert, use_chinese=opt.chinese)
            testset = ABSADataset(opt.dataset_file['test'], tokenizer, max_seq_len=max_seq_len, use_bert=opt.use_bert, use_chinese=opt.chinese)
            self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
            self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=opt.device.index))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        writer = SummaryWriter(log_dir=self.opt.logdir)
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                if self.opt.use_bert:
                    inputs = [sample_batched[col] for col in self.opt.inputs_cols]
                else:
                    inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = 'state_dict/{0}_{1}_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(test_acc, 4))
                            torch.save(self.model.state_dict(), path)
                            print('>> saved: ' + path)
                    if f1 > max_f1:
                        max_f1 = f1

                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('acc', train_acc, global_step)
                    writer.add_scalar('test_acc', test_acc, global_step)
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))

        writer.close()
        return max_test_acc, max_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                if self.opt.use_bert:
                    t_inputs = [t_sample_batched[col] for col in self.opt.inputs_cols]
                else:
                    t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                #t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            print('repeat: ', i)
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, optimizer, max_test_acc_overall=max_test_acc_overall)
            print('max_test_acc: {0}     max_f1: {1}'.format(max_test_acc, max_f1))
            if max_test_acc > max_test_acc_overall:
                max_test_acc_overall = max_test_acc
                torch.save(self.model.state_dict(),
                           os.path.join(os.getcwd(), 'nsh_absa_best_model.pkl'))
            max_f1_overall = max(max_f1, max_f1_overall)
            print('#' * 100)
        print("max_test_acc_overall:", max_test_acc_overall)
        print("max_f1_overall:", max_f1_overall)


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aoa', type=str)
    parser.add_argument('--dataset', default='yuqing', type=str, help='twitter, restaurant, laptop, yuqing')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--use_bert', default=False, type=bool)
    parser.add_argument('--chinese', default=True, type=str)
    parser.add_argument('--model_path', default='./bert_pretrain_model/uncased_L-12_H-768_A-12/', type=str)
    opt = parser.parse_args()

    if opt.use_bert:
        opt.hidden_dim = 768
        opt.embed_dim = 768
    if opt.chinese:
        jieba.load_userdict('nsh_dict.txt')
        opt.model_path = 'nishuihan_word_embedding.bin'
        opt.fasttext_word_embedding_model = fasttext.load_model(opt.model_path)
        opt.embed_dim = opt.fasttext_word_embedding_model.dim
        opt.max_seq_len = 35

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_aoa': Bert_BaseLine,
        'bert_mgan': Bert_MGAN
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'yuqing': {
            'train': './datasets/yuqing/train.txt',
            'test': './datasets/yuqing/test.txt'
        }
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_aoa': ['text_raw_indices', 'text_raw_indices_mask', 'aspect_in_text'],
        'bert_mgan': ['text_raw_indices', 'text_raw_indices_mask', 'aspect_in_text']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run(5)
