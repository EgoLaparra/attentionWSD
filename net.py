#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:15:56 2016

@author: egoitz
"""
import theano
import theano.tensor as T

import nn
import optimizers
import functions

class Model():
    
    def __init__(self, rng, vocab_size, bs,
                ctx_emb_units=50, tgt_emb_units=100,
                batches=T.matrices('batches',dtype='int64'),
                target=T.matrix('target',dtype='int64'),
                context=T.matrix('context',dtype='int64'),
                wn=T.matrix('wn',dtype='float64')):
        

        # MODEL
        self.ctx_emb = nn.Embedding(rng, context, vocab_size, ctx_emb_units, sequence=True)
        self.lstm = nn.LSTM(rng, self.ctx_emb.output, ctx_emb_units, tgt_emb_units, peephole_output=True)
        self.tgt_emb = nn.Embedding(rng, target, vocab_size, tgt_emb_units, sequence=True)
        self.query = nn.Combine([self.lstm.output, self.tgt_emb.output], op='mean')
        self.top = nn.Attention(self.query.output, wn)

        # PARAMETERS TO BE LEARNT
        self.params = self.ctx_emb.params + self.lstm.params + self.tgt_emb.params
        self.deltas = self.ctx_emb.deltas + self.lstm.deltas + self.tgt_emb.deltas

        # LOSS FUNCTION, COSINE SIMILARITY
        def sentence_cosine(indexes, output):
            return functions.multi_cosine(output[indexes])
        cosines = theano.scan(fn=sentence_cosine,
                              outputs_info=None,
                              sequences=[batches],
                              non_sequences=self.top.output)
        self.loss = T.sum(cosines)

        # BACK-PROPAGATION, GET GRADIENTS
        self.paramsGrad = [T.grad(self.ce, param) for param in self.params]

        # UPDATE WEIGHTS AND BIASES
        self.optimizer = optimizers.SGD(self.deltas, self.params, self.paramsGrad, batch_size=bs)
        self.updates = self.optimizer.updates()
