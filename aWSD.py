#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 20:35:41 2016

@author: egoitz
"""

import sys
import data
import numpy as np
from numpy import random
import theano
import theano.tensor as T

import nn
import optimizers

sys.path.append('/home/egoitz/Tools/embeddings/kge/SME/')

# PARAMETERS
dataset_file = sys.argv[1]
sequence_size = 4
epochs = 1

# HYPERPARAMETERS
batch_size = 100 # s
learning_rate = .1
momentum = .9
embedding_units = 50 # e
hidden_units = 200 # h

hidden_units = 60 # h

init_sd = .01 # Standard deviation of the normal distribution used for weight initializations
                                                                                                                                                                         
# VARIABLES FOR TRACKING TRAINING PROGRESS                                                                                                                             
show_training_ce_after = 100
show_validation_ce_after = 1000
    
# LOAD DATA
print ('Loading knowledge graph embeddings...')
wn_kge, wn_dict = data.load_kge()
print ('Loading data...')
(vocab, train_set, test_set, valid_set) = data.load_file(dataset_file, sequence_size)
vocab_size = len(vocab) # v
(train_input, train_target,
 test_input, test_target,
 valid_input, valid_target) = data.load_data(train_set, test_set, valid_set, batch_size)
(batches, batch_size, words_num) = np.shape(train_input) # b, s, w

sparse_matrix = T.eye(vocab_size) # V x V
n_sparse_matrix = np.eye(vocab_size)
count = 0
tiny = np.exp(-30)
t = theano.shared(np.exp(-30)) # Avoids divide by zero in log

# INITIALIZE WEIGHTS AND BIASES
x = T.matrix('x',dtype='int64')
y = T.vector('y',dtype='int64')
(bs, wn) = T.shape(x)
s_y = sparse_matrix[y,:]

rng = random.RandomState(12345)
emb = nn.Embedding(rng, x, vocab_size, embedding_units, batch_size=bs, sequence=True)
lstm = nn.LSTM(rng, emb.output, embedding_units, hidden_units, batch_size=bs, peephole_output=True)
top = nn.Dense(rng, lstm.output, hidden_units, vocab_size, activation=T.nnet.softmax, batch_size=bs)

params = emb.params + lstm.params + top.params
deltas = emb.deltas + lstm.deltas + top.deltas

# LOSS FUNCTION, CROSS-ENTROPY
ce = T.sum(T.nnet.nnet.categorical_crossentropy((top.output + t), s_y))

# BACK-PROPAGATION, GET GRADIENTS
paramsGrad = [T.grad(ce, param) for param in params]

# UPDATE WEIGHTS AND BIASES
optimizer = optimizers.SGD(deltas, params, paramsGrad,batch_size=bs)
updates = optimizer.updates()
