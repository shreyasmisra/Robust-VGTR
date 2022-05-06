# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
                 input_dropout_p=0., dropout_p=0., n_layers=2, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1
        # self._init_param()

    def _init_param(self):
        for k, v in self.rnn.named_parameters():
            if 'bias' in k:
                v.data.zero_().add_(1.0)  # init LSTM bias = 1.0

    def forward(self, input_labels):

        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1)
            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()
            s2r = {s: r for r, s in enumerate(sort_ixs)}  # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]
            assert max(input_lengths_list) == input_labels.size(1)

            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            input_labels = input_labels[sort_ixs]

        embedded = self.embedding(input_labels)
        embedded = self.input_dropout(embedded)
        embedded = self.mlp(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        output, hidden = self.rnn(embedded)

        if self.variable_lengths:

            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            if self.rnn_type == 'lstm':
                hidden = hidden[0]
            hidden = hidden[:, recover_ixs, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)

        return output, hidden, embedded



class PhraseAttention(nn.Module):
    def __init__(self, input_dim):
        super(PhraseAttention, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, input_labels):

        cxt_scores = self.fc(context).squeeze(2) # B, seq len
        attn = F.softmax(cxt_scores)

        #is_not_zero = (input_labels != 0).float()
        #attn = attn * is_not_zero
        #attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)
        weighted_emb = torch.bmm(attn3, embedded)
        weighted_emb = weighted_emb.squeeze(1)

        return attn, weighted_emb


class TextualEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding=True)
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.fc1 = nn.Linear(768,256)
        #self.rnn = RNNEncoder(args.vocab_size, args.embedding_dim,
         #             args.hidden_dim, args.rnn_hidden_dim,
          #            bidirectional=True,
          #            input_dropout_p=0.1,
          #            dropout_p=0.1,
          #            n_layers=args.rnn_layers,
          #            variable_lengths=True,
          #            rnn_type='lstm')
        self.parser = nn.ModuleList([PhraseAttention(input_dim=args.rnn_hidden_dim * 2)
                       for _ in range(args.num_exp_tokens)])

    def forward(self, sent):
    # sent -> phrase
        #max_len = (sent != 0).sum(1).max().item()
        #sent = sent[:, :max_len]
        tokenized_inp = self.tokenizer(sent, return_tensors="pt", padding=True)
        
        input_ids = tokenized_inp['input_ids'].cuda()
        token_type_ids = tokenized_inp['token_type_ids'].cuda()
        attention_mask = tokenized_inp['attention_mask'].cuda()
        
        output = self.bert_model(input_ids, attention_mask, token_type_ids)
        
        context, hidden, embed = output.last_hidden_state, output.hidden_states[0], output.hidden_states[1]
        
        hidden = hidden.detach().cpu()
        embed = embed.detach().cpu()
        input_ids = input_ids.detach().cpu()
        token_type_ids = token_type_ids.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        #output_vec = self.fc1(context)
       
        #context, hidden, embedded = self.rnn(sent)  # [bs, maxL, d]
        # sent_feature = [module(context, embedded, sent)[-1] for module in self.parser]
        #print(hidden.shape, embed.shape, context.shape)
        
        #sent_feature = [module(context, embed, sent)[-1] for module in self.parser]
        #return torch.stack(sent_feature, dim=1)
        return context


def build_textual_encoder(args):
    return TextualEncoder(args)

