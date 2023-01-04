def display_sequence(x):

    for word in x:
        if word == '<PAD>':
            continue
        sys.stdout.write(" " + word)
    sys.stdout.write("\n")
    sys.stdout.flush()

def display_sequence_with_pos(x, p):

    for word, pos in zip(x, p):
#         print(word, pos)
        if word == '<PAD>':
            continue
        if pos == 0:
            sys.stdout.write(" " + word)
        else:
            sys.stdout.write(" " + '[[' + word + ']]')
    sys.stdout.write("\n")
    sys.stdout.flush()

def get_multiview_multidim_examples_pkl(fpath, max_seq_len=300, max_sent_num=200, bpe_token=False):
    """
    Get data from tsv files. 
    Input:
        fpath -- the file path.
        Assume number of classes = 2
        
    Output:
        ts -- a list of strings (each contain the text)
        ys -- float32 np array (num_example, )
        ss -- float32 np array (num_example, num_sent, sequence_length)
    """
    MBTI_LABELS = [['I', 'E'], ['N', 'S'], ['T', 'F'], ['J', 'P']]
    
    n = 0
    num_empty = 0
    
    train_ds = [[], [], [], []]
    train_ss = [[], [], [], []]
    train_ps = [[], [], [], []]
    train_ys = [[], [], [], []]
    dev_ds = [[], [], [], []]
    dev_ss = [[], [], [], []]
    dev_ps = [[], [], [], []]
    dev_ys = [[], [], [], []]
    test_ds = [[], [], [], []]
    test_ss = [[], [], [], []]
    test_ps = [[], [], [], []]
    test_ys = [[], [], [], []]
    
    real_max_sent_num = 0
    real_max_len = 0
    total_len = 0
    
    label_dict = {}

    with open(fpath, "rb") as fh:
        char_list = pickle.load(fh)
        fh.close()
    
    loaded_inst_dict = {}
    with open(fpath, "r", encoding='cp1256') as f:
        for inst_id, instance in tqdm(enumerate(char_list)):

            sections = instance['dialog_text']
            
            if len(sections) == 0:
                text = ''
            else:
                if bpe_token:
                    text = ' '.join(sections)
                else:
                    text = ' '.join(sections).lower()

            t = text.split()
            if len(t) > real_max_len:
                real_max_len = len(t)
                
            t = t[:max_seq_len]
            
            sections = instance['scene_text']
#             sections = instance['mention_text']
            s = []
            positions = []
            for section in sections:
                sec_title_lower_old = section[0].lower()
                sec_title_lower = section[1].lower()
                sec_title = section[1].lower().split()
                sec_text = section[2].lower().split()
                
                inst_text = []
                inst_pos = []
                
                tid = 0
                while tid < len(sec_text):
                    token = sec_text[tid]
                    if len(sec_text) - tid < len(sec_title):
                        inst_text.append(token)
                        inst_pos.append(0)
                        tid += 1
                    else:
                        if ' '.join(sec_text[tid:tid+len(sec_title)]) == sec_title_lower or                                   ' '.join(sec_text[tid:tid+len(sec_title)]) == sec_title_lower_old:
                            for k in range(tid, tid+len(sec_title)):
                                inst_text.append(sec_text[k])
                                inst_pos.append(1)
                            tid = tid+len(sec_title)
                        else:
                            inst_text.append(token)
                            inst_pos.append(0)
                            tid += 1
                s += inst_text
                positions += inst_pos

#             s = []
#             positions = []
            if len(s) > real_max_len:
                real_max_len = len(s)
                
            s = s[:max_seq_len]
            p = positions[:max_seq_len]
                
            if len(t) == 0:
                t.append('EMPTY')
#                 num_empty += 1
#                 print('empty instance:', instance['subcategory'] + '\t' + instance['mbti_profile'])
#                 continue

            if len(s) == 0:
                s.append('EMPTY')
                p.append(0)

            total_len += len(t) + len(s)
            tokens = []
            labels = [-1, -1, -1, -1]

            for y in instance.keys():
                for dim in range(4):
                    if y in MBTI_LABELS[dim] and not math.isnan(instance[y]):
                        label = MBTI_LABELS[dim].index(y)
                        labels[dim] = label
                    
            
            for dim in range(4):
                if labels[dim] == -1:
#                     print(instance)
                    continue
                if inst_id < 2166:
                    train_ds[dim].append(t)
                    train_ss[dim].append(s)
                    train_ps[dim].append(p)
                    train_ys[dim].append(labels[dim])
                    
                    y = MBTI_LABELS[dim][labels[dim]]
                    if y not in label_dict:
                        label_dict[y] = 1
                    else:
                        label_dict[y] += 1
                    
                elif inst_id < 2666:
                    dev_ds[dim].append(t)
                    dev_ss[dim].append(s)
                    dev_ps[dim].append(p)
                    dev_ys[dim].append(labels[dim])
                else:
                    test_ds[dim].append(t)
                    test_ss[dim].append(s)
                    test_ps[dim].append(p)
                    test_ys[dim].append(labels[dim])

            n += 1
    
    print("Number of examples: %d" % n)
    print("Number of empty examples: %d" % num_empty)
    for dim in range(4):
        print('Dimension %d:'%dim)
        print("Number of training examples: %d" % len(train_ds[dim]))
        print("Number of dev examples: %d" % len(dev_ds[dim]))
        print("Number of test examples: %d" % len(test_ds[dim]))
    print("Maximum length: %d" % real_max_len)
    print("Average length: %d" % (total_len / n))

    print(label_dict)
    
    ret_list = [(train_ds, train_ss, train_ps, train_ys)]
    ret_list.append((dev_ds, dev_ss, dev_ps, dev_ys))
    ret_list.append((test_ds, test_ss, test_ps, test_ys))

    return ret_list, label_dict

def get_mbti_multiview_multirow_multidim_datasets_pkl(data_dir, max_seq_len=300, max_sent_num=10, word_thres=10):
    """
    Get datasets (train, dev and test).
    """
    
    ##### load data from file
#     ret_list = [(train_ds, train_ss, train_ps, train_ys)]
#     ret_list.append((dev_ds, dev_ss, dev_ps, dev_ys))
#     ret_list.append((test_ds, test_ss, test_ps, test_ys))

    ret_list, label_dict = get_multiview_multidim_examples_pkl(data_dir,
                              max_seq_len=max_seq_len*max_sent_num, max_sent_num=max_sent_num)
    train_tuple = ret_list[0]
    dev_tuple = ret_list[1]
    test_tuple = ret_list[2]
    
#     t_tr, y_tr, t_d, y_d, t_t, y_t, label_dict = get_multidim_examples_pkl(data_dir, 
#                               max_seq_len=max_seq_len*max_sent_num, max_sent_num=max_sent_num)

    print(label_dict)
    ##### construct torch datasets
    NUM_DIM = 4
    
    D_tr_list = []
    D_dev_list = []
    D_test_list = []
    for dim in range(NUM_DIM):
        dim_tuple = [train_tuple[i][dim] for i in range(4)]
        D_tr = BertMBTIMultiviewMultirowDataset(dim_tuple, max_seq_len, max_sent_num)
        dim_tuple = [dev_tuple[i][dim] for i in range(4)]
        D_dev = BertMBTIMultiviewMultirowDataset(dim_tuple, max_seq_len, max_sent_num)
        dim_tuple = [test_tuple[i][dim] for i in range(4)]
        D_test = BertMBTIMultiviewMultirowDataset(dim_tuple, max_seq_len, max_sent_num)
        
        D_tr_list.append(D_tr)
        D_dev_list.append(D_dev)
        D_test_list.append(D_test)
#     D_te = BeerDataset(te_outputs, vocab.stoi, max_seq_len, max_sent_num, eval_set=True)
    
    return D_tr_list, D_dev_list, D_test_list, label_dict #, D_te


# D_tr_list, D_dev_list, D_test_list, label_dict = get_mbti_multiview_multirow_multidim_datasets_pkl(
#                                             'add_Big5.tok.bert_tok.pkl', 
#                                              max_seq_len=100, max_sent_num=10)

import os
import sys
import math
import csv
import collections
import numpy as np
import gzip
import json

# import pickle5 as pickle
import pickle

import torch
from torchtext.vocab import Vocab
from torch.utils.data import Dataset

from tqdm import tqdm

import transformers
from transformers import BertTokenizer, GPT2Tokenizer

print(transformers.__version__)

class BertMBTIMultiviewMultirowDataset(Dataset):
    """Beer dataset."""

    def __init__(self, data, max_seq_len, max_sent_num=10, eval_set=False, transform=None):
        self.data = data
#         self.stoi = stoi
        self.max_seq_len = max_seq_len
        self.max_sent_num = max_sent_num
        self.transform = transform
        self.eval_set = eval_set
        
        self.CLS_TOKEN = 101
        self.SEP_TOKEN = 102
        self.BEGIN_ENTITY_TOKEN = 104
        self.END_ENTITY_TOKEN = 105
        self.SPLIT_TOKEN = 5
        self.PAD_TOKEN = 0
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        
        texts, scenes, positions, ys = self.data
                
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = texts[idx]

        if len(text) > self.max_seq_len * self.max_sent_num:
            text = text[0:self.max_seq_len*self.max_sent_num]
            
        text = self.tokenizer.convert_tokens_to_ids(text)
        mask = [1.] * len(text)

        num_rows = (len(text) // self.max_seq_len)
        if num_rows * self.max_seq_len < len(text):
            num_rows += 1
        while len(text) < num_rows * self.max_seq_len:
            text.append(self.PAD_TOKEN)
            mask.append(0.)
            
        xs = []
        ms = []
        
        for row_id in range(num_rows):
            x = text[row_id*self.max_seq_len:(row_id+1)*self.max_seq_len]
            m = mask[row_id*self.max_seq_len:(row_id+1)*self.max_seq_len]
            x = [self.CLS_TOKEN] + x + [self.SEP_TOKEN]
            if m[-1] == 1.0:
                m = [1.0] + m + [1.0]
            else:
                m = [1.0] + m + [0.0]
                
            xs.append(x)
            ms.append(m)

        assert len(xs[0]) == self.max_seq_len + 2, str(len(xs[0]))
        assert len(ms[0]) == self.max_seq_len + 2, str(len(ms[0]))
        
        
        scene = scenes[idx]
        pos = positions[idx]

        if len(scene) > self.max_seq_len * self.max_sent_num:
            scene = scene[0:self.max_seq_len*self.max_sent_num]
            pos = pos[0:self.max_seq_len*self.max_sent_num]
            
        scene = self.tokenizer.convert_tokens_to_ids(scene)
        mask = [1.] * len(scene)

        num_rows = (len(scene) // self.max_seq_len)
        if num_rows * self.max_seq_len < len(scene):
            num_rows += 1
        while len(scene) < num_rows * self.max_seq_len:
            scene.append(self.PAD_TOKEN)
            pos.append(0.)
            mask.append(0.)
            
        ss = []
        sms = []
        ps = []
        
        for row_id in range(num_rows):
            new_x = []
            new_m = []
            new_p = []
            for tok_id in range(row_id*self.max_seq_len, (row_id+1)*self.max_seq_len):
                if scene[tok_id] == self.PAD_TOKEN:
                    break
                    
                # represent target entities as replacements
                if pos[tok_id] == 1.0 and (tok_id == row_id*self.max_seq_len or pos[tok_id - 1] == 0.0):
                    new_x.append(self.BEGIN_ENTITY_TOKEN)
                    new_m.append(1.0)
                    new_p.append(1.0)
                elif pos[tok_id] == 1.0:
                    continue
                else: # pos[tok_id] == 0.0
                    new_x.append(scene[tok_id])
                    new_m.append(mask[tok_id])
                    new_p.append(0.0)
            
            new_x = [self.CLS_TOKEN] + new_x[:self.max_seq_len] + [self.SEP_TOKEN]
            new_p = [0.0] + new_p[:self.max_seq_len] + [0.0]
            if new_m[-1] == 1.0:
                new_m = [1.0] + new_m[:self.max_seq_len] + [1.0]
            else:
                new_m = [1.0] + new_m[:self.max_seq_len] + [0.0]
                
            while len(new_x) < self.max_seq_len + 2:
                new_x.append(self.PAD_TOKEN)
                new_m.append(0.0)
                new_p.append(0.0)
                
            ss.append(new_x)
            ps.append(new_p)
            sms.append(new_m)

        assert len(xs[0]) == self.max_seq_len + 2, str(len(xs[0]))
        assert len(ps[0]) == self.max_seq_len + 2, str(len(ps[0]))
        assert len(ms[0]) == self.max_seq_len + 2, str(len(ms[0]))
        
#         print(idx, np.array(xs, dtype=np.int64).shape, np.array(ms, dtype=np.int64).shape)
        
        sample = {"x": np.array(xs, dtype=np.int64), 
                  "mask_x": np.array(ms, dtype=np.float32),
                  "s": np.array(ss, dtype=np.int64), 
                  "pos": np.array(ps, dtype=np.float32),
                  "mask_s": np.array(sms, dtype=np.float32),
                  "y":  np.array(ys[idx], dtype=np.int64)
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This is sentence level Tao's Model.
"""

import os
import random
import numpy as np 
import torch
import sys

# set seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from collections import deque

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

# from movie_mbti_data_multiview_bert_utils import display_sequence, display_sequence_with_pos
# from movie_mbti_data_multiview_bert_utils import get_mbti_multiview_multirow_multidim_datasets_pkl

from transformers import BertModel, BertTokenizer
import logging

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[ ]:




D_tr_list, D_dev_list, D_test_list, label_dict = get_mbti_multiview_multirow_multidim_datasets_pkl('MVMR_BERT_tok.pkl', 
                                             max_seq_len=100, max_sent_num=5, word_thres=1)#2k 5,10,20,40


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[ ]:


# if y == 'i' or y == 'n' or y == 't' or y == 'j':
#     labels.append(1)
# print(5371 + 3881)
weight_ie = [label_dict['E'] / (label_dict['E'] + label_dict['I']), 
             label_dict['I'] / (label_dict['E'] + label_dict['I'])]
weight_ns = [label_dict['S'] / (label_dict['S'] + label_dict['N']), 
             label_dict['N'] / (label_dict['S'] + label_dict['N'])]
weight_tf = [label_dict['F'] / (label_dict['F'] + label_dict['T']), 
             label_dict['T'] / (label_dict['F'] + label_dict['T'])]
weight_jp = [label_dict['P'] / (label_dict['P'] + label_dict['J']), 
             label_dict['J'] / (label_dict['P'] + label_dict['J'])]

MBTI_LABELS = [['I', 'E'], ['N', 'S'], ['T', 'F'], ['J', 'P']]

label_weights = [weight_ie, weight_ns, weight_tf, weight_jp]

print(weight_ie)
print(weight_ns)
print(weight_tf)
print(weight_jp)


# In[ ]:


print(len(D_dev_list[0].data[0][0]))
print(len(D_dev_list[0].data[0]))
display_sequence(D_dev_list[0].data[0][0])
display_sequence(D_dev_list[0].data[1][0])
display_sequence_with_pos(D_dev_list[0].data[1][0], D_dev_list[0].data[2][0])


# In[ ]:



class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.2):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class BertMultiViewMultiDimClassifier(nn.Module):
    def __init__(self, args, max_length, blank_padding=True):
        super().__init__()
        
#         args.hidden = args.hidden_dim
        num_class = args.num_classes
        
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        
        self.num_views = 2
        
        self.attentive_bert = True

        logging.info('Loading BERT pre-trained checkpoint.')
#         self.bert = BertModel.from_pretrained(pretrain_path)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Token-level rationale classifier (serve as a head)
        self.attention_fc_ie = nn.Linear(self.hidden_size, 1, bias=False)
        self.attention_fc_ns = nn.Linear(self.hidden_size, 1, bias=False)
        self.attention_fc_tf = nn.Linear(self.hidden_size, 1, bias=False)
        self.attention_fc_jp = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.attention_fcs=[self.attention_fc_ie, self.attention_fc_ns, 
                            self.attention_fc_tf, self.attention_fc_jp]
        
        self.s_attention_fc_ie = nn.Linear(self.hidden_size, 1, bias=False)
        self.s_attention_fc_ns = nn.Linear(self.hidden_size, 1, bias=False)
        self.s_attention_fc_tf = nn.Linear(self.hidden_size, 1, bias=False)
        self.s_attention_fc_jp = nn.Linear(self.hidden_size, 1, bias=False)
        
        self.s_attention_fcs=[self.s_attention_fc_ie, self.s_attention_fc_ns, 
                            self.s_attention_fc_tf, self.s_attention_fc_jp]

        
        # Task predictor 
        self.pred_fc_ie = nn.Linear(self.hidden_size*self.num_views, args.num_classes)
        self.pred_fc_ns = nn.Linear(self.hidden_size*self.num_views, args.num_classes)
        self.pred_fc_tf = nn.Linear(self.hidden_size*self.num_views, args.num_classes)
        self.pred_fc_jp = nn.Linear(self.hidden_size*self.num_views, args.num_classes)
        
        self.pred_fcs = [self.pred_fc_ie, self.pred_fc_ns, self.pred_fc_tf, self.pred_fc_jp]
        
#         self.loss = nn.CrossEntropyLoss()
        
    def pred_vars(self):
        """
        Return the variables of the predictor.
        """
        params = list(
            self.bert.parameters()) + list(
            self.pred_fc_ie.parameters()) + list(
            self.pred_fc_ns.parameters()) + list(
            self.pred_fc_tf.parameters()) + list(
            self.pred_fc_jp.parameters()) + list(
            self.attention_fc_ie.parameters()) + list(
            self.attention_fc_ns.parameters()) + list(
            self.attention_fc_tf.parameters()) + list(
            self.attention_fc_jp.parameters()) + list(
            self.s_attention_fc_ie.parameters()) + list(
            self.s_attention_fc_ns.parameters()) + list(
            self.s_attention_fc_tf.parameters()) + list(
            self.s_attention_fc_jp.parameters())
        
        return params
    
    def forward(self, x, mask_x, s, p, mask_s, dim):
        """
        Args:
            seq: (R, L), index of tokens
            mask: (R, L), index of action tokens
        """
        LARGE_NEG = -1e9
        mask_x_ = mask_x.view(-1).unsqueeze(-1) # (num_row*seq_length, 1)
        p_ = p.view(-1).unsqueeze(-1) # (num_row*seq_length, 1)
        mask_s_ = mask_s.view(-1).unsqueeze(-1) # (num_row*seq_length, 1)
        
        x_size = x.size(0)
        s_size = s.size(0)
        
#         all_input = torch.cat([x, s], dim=0)
#         all_mask = torch.cat([mask_x, mask_s], dim=0)
        
#         all_hiddens = self.bert(all_input, attention_mask=all_mask)
#         all_hiddens = all_hiddens[0]
        
#         (hiddens, s_hiddens) = torch.split(all_hiddens, [x_size, s_size], dim=0)
#         hiddens_ = hiddens.view(-1, hiddens.size(2)) # (R*L, H)
#         s_hiddens_ = s_hiddens.view(-1, s_hiddens.size(2)) # (R*L, H)
        
        if x_size != 0:
            hiddens = self.bert(x, attention_mask=mask_x)
            hiddens = hiddens[0]
            hiddens_ = hiddens.view(-1, hiddens.size(2)) # (R*L, H)
            
        if s_size != 0:
            s_hiddens = self.bert(s, attention_mask=mask_s) # (R, L, H)
            s_hiddens = s_hiddens[0]
            s_hiddens_ = s_hiddens.view(-1, s_hiddens.size(2)) # (R*L, H)
        
        if self.attentive_bert:
            # (R, L) 
            token_att_logits = self.attention_fcs[dim](hiddens).squeeze(-1)
            tmp_logits = token_att_logits + (1.-mask_x)*LARGE_NEG # (R, L)
            
            token_probs = F.softmax(tmp_logits.view(-1), dim=0).unsqueeze(-1) #(R*L, 1)

            dialog_state = torch.sum(hiddens_ * token_probs * mask_x_, dim=0).unsqueeze(0) # (1, H)

            # (R, L) 
            s_token_att_logits = self.s_attention_fcs[dim](s_hiddens).squeeze(-1)
            s_tmp_logits = s_token_att_logits + (1.-p)*LARGE_NEG # (R, L)
            
            s_token_probs = F.softmax(s_tmp_logits.view(-1), dim=0).unsqueeze(-1) #(R*L, 1)
            if torch.sum(p) == 0:
                s_token_probs = torch.zeros_like(s_token_probs)
                
            scene_state = torch.sum(s_hiddens_ * s_token_probs * p_, dim=0).unsqueeze(0) # (1, H)

#             print(dialog_state.size())
#             print(scene_state.size())
#             print(torch.cat([dialog_state, scene_state], dim=1).size())
            # classification
            pred_logits = self.pred_fcs[dim](torch.cat([dialog_state, scene_state], dim=1)).squeeze(0) # (2,)

        return pred_logits


# In[ ]:


class Dummy():
    pass

args = Dummy()
args.rnn_dim = 32
args.num_classes = 2
args.gumbel_temp = 1.0 # 0.1

args.lambda_smooth = 1e-3

args.exploration_rate = 0.2
args.highlight_ratio = 0.1

args.dropout = 0.2

# args.highlight_ratio = 0.05

args.l2_decay = 1e-4

model = BertMultiViewMultiDimClassifier(args, max_length=512).cuda()
print(model)


# In[ ]:


from tqdm import tqdm

num_epochs = 20
batch_size = 1
gradient_accumulation_steps = 10

D_tr_list_ = []
D_dev_list_ = []
D_test_list_ = []

for dim in range(4):
    D_tr_ = DataLoader(D_tr_list[dim], batch_size=batch_size, shuffle=True) #, num_workers=16)
    D_dev_ = DataLoader(D_dev_list[dim], batch_size=batch_size, shuffle=False) #, num_workers=4)
    D_test_ = DataLoader(D_test_list[dim], batch_size=batch_size, shuffle=False) #, num_workers=4)
    
    D_tr_list_.append(D_tr_)
    D_dev_list_.append(D_dev_)
    D_test_list_.append(D_test_)


# In[ ]:



pred_optimizer = torch.optim.Adam(model.pred_vars() , lr=2e-5)
best_F1s = [0., 0., 0., 0.]

counter = 0

# switch_epoch = 0

# queue_length = 200
# history_rewards = deque(maxlen=queue_length)
# history_rewards.append(0.)

# TODO: adding gradient accumulation

for i_epoch in range(num_epochs):

    print ("================")
    print ("epoch: %d" % i_epoch)
    print ("================")

    model.train()
    train_accs = [[], [], [], []]
    step = 0
    pred_optimizer.zero_grad()
    
    for i_batch in tqdm(range(len(D_tr_list[0].data[0]) // batch_size)):
#     for i_batch, data in enumerate(D_tr_):  
#         print(i_batch)
        for dim in range(4):
            data = next(iter(D_tr_list_[dim]))            
            counter += 1

            x = data["x"].cuda().squeeze(0)
            mask_x = data["mask_x"].cuda().squeeze(0)
            s = data["s"].cuda().squeeze(0)
            mask_s = data["mask_s"].cuda().squeeze(0)
            p = data["pos"].cuda().squeeze(0)
            y = data["y"].cuda()
            sent_mask = None
            if i_batch==0 and dim == 0 and counter == 1:
                print(x.size())

            # update the classifiers
            pred_logits = model(x, mask_x, s, p, mask_s, dim).unsqueeze(0)
        
            tmp_loss = F.cross_entropy(pred_logits, y, reduction='none') # (batch_size,)
            weights = (y == 0).int() * 2 * label_weights[dim][0] + (y == 1).int() * 2 * label_weights[dim][1]
            sup_loss = torch.mean(tmp_loss * weights) / gradient_accumulation_steps
#             sup_loss = F.cross_entropy(pred_logits, y)

            if step == 0:
                accum_loss = sup_loss.cpu().item()
            else:
                accum_loss += sup_loss.cpu().item()

            sup_loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                pred_optimizer.step()
                pred_optimizer.zero_grad()
                step = 0
                
            step += 1

            pred = torch.max(pred_logits, 1)[1]
            acc = (pred == y).sum().float() / y.shape[0]
            train_accs[dim].append(acc.cpu().item())
        
        if (i_batch + 1) % 200 == 0:
            for dim in range(4): 
                print ("Dim %d Train batch: %d, sup loss: %.4f acc: %.4f" % (dim, i_batch, accum_loss, 
                                                                         np.mean(train_accs[dim])))
            sys.stdout.flush()
            train_accs = [[], [], [], []]
#     sys.stdout.flush()
    
        if (i_batch + 1) % 1100 == 0 or (i_batch+1) == (len(D_tr_list[0].data[0]) // batch_size):
    
            model.eval()

            with torch.no_grad():
                for mbti_dim in range(4):

                    preds = [1e-9, 1e-9]
                    gts = [0., 0.]
                    corrects = [1e-9, 1e-9]

                    num_sample = 0

                    for i_batch, data in enumerate(D_dev_list_[mbti_dim]):
                        x = data["x"].cuda().squeeze(0)
                        mask_x = data["mask_x"].cuda().squeeze(0)
                        s = data["s"].cuda().squeeze(0)
                        mask_s = data["mask_s"].cuda().squeeze(0)
                        p = data["pos"].cuda().squeeze(0)
                        y = data["y"].cuda()
                        sent_mask = None

                        pred_logits = model(x, mask_x, s, p, mask_s, mbti_dim).unsqueeze(0)

                        pred = torch.max(pred_logits, 1)[1]
                        for label_id in [0,1]:
                            preds[label_id] += (pred == label_id).sum().float().item()
                            gts[label_id] += (y == label_id).sum().float().item()
                            corrects[label_id] += ((pred == y).long() * (pred == label_id).long()).sum().float().item()

                        num_sample += y.shape[0]

                    avg_f1 = 0.
                    print('Dev {}'.format('/'.join(MBTI_LABELS[mbti_dim])))
                    for label_id in [0,1]:
                        prec = corrects[label_id] / preds[label_id]
                        rec = corrects[label_id] / gts[label_id]
                        f1 = 2 * prec * rec / (prec + rec)
                        avg_f1 += f1
                        print('label %d prec: %.4f rec: %.4f f1:%.4f'%(label_id, prec, rec, f1))
                    print('%s Dev Macro F1: %.4f'%('/'.join(MBTI_LABELS[mbti_dim]), avg_f1 / 2))

                    if avg_f1 / 2 > best_F1s[mbti_dim]:
                        best_F1s[mbti_dim] = avg_f1 / 2
                    print('Best %s Dev Macro F1: %.4f'%('/'.join(MBTI_LABELS[mbti_dim]), best_F1s[mbti_dim]))

                    preds = [1e-9, 1e-9]
                    gts = [0., 0.]
                    corrects = [1e-9, 1e-9]

                    num_sample = 0

                    for i_batch, data in enumerate(D_test_list_[mbti_dim]):
                        x = data["x"].cuda().squeeze(0)
                        mask_x = data["mask_x"].cuda().squeeze(0)
                        s = data["s"].cuda().squeeze(0)
                        mask_s = data["mask_s"].cuda().squeeze(0)
                        p = data["pos"].cuda().squeeze(0)
                        y = data["y"].cuda()
                        sent_mask = None

                        pred_logits = model(x, mask_x, s, p, mask_s, mbti_dim).unsqueeze(0)

                        pred = torch.max(pred_logits, 1)[1]
                        for label_id in [0,1]:
                            preds[label_id] += (pred == label_id).sum().float().item()
                            gts[label_id] += (y == label_id).sum().float().item()
                            corrects[label_id] += ((pred == y).long() * (pred == label_id).long()).sum().float().item()

                        num_sample += y.shape[0]

                    avg_f1 = 0.
                    print('Test {}'.format('/'.join(MBTI_LABELS[mbti_dim])))
                    for label_id in [0,1]:
                        prec = corrects[label_id] / preds[label_id]
                        rec = corrects[label_id] / gts[label_id]
                        f1 = 2 * prec * rec / (prec + rec)
                        avg_f1 += f1
                        print('label %d prec: %.4f rec: %.4f f1:%.4f'%(label_id, prec, rec, f1))
                    print('%s Test Macro F1: %.4f'%('/'.join(MBTI_LABELS[mbti_dim]), avg_f1 / 2))

            sys.stdout.flush()
            model.train()
            


# In[ ]:


# a, b = model.bert(x, attention_mask=mask)
# print(a)
# print(b)


# In[ ]:





# In[ ]:


# data = next(iter(D_tr_list_[dim]))

# x=data['x'].squeeze(0)
# mask=data['mask'].squeeze(0)
# print(y.size())
print(x.size())
print(mask_x.size())
print(s.size())
print(mask_s.size())
# print(pred_logits.size())
# print(pred_logits)
# print(y)


