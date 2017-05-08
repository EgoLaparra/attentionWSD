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
    
    def __init__(self, rng, vocab_size, lemma_inv_size,
                ctx_emb_units=50, tgt_emb_units=20,
                targets=None, contexts=None, context_mask=None,
                senses=None, senses_mask=None, case_indexes=T.matrix(dtype='int64')
                ):
        
        bs = T.shape(targets)[0]
        
        # MODEL
        self.ctx_emb = nn.Embedding(rng, contexts, vocab_size, ctx_emb_units, sequence=True) # output: num_words X batch_size x ctx_emb_size
        self.lstm = nn.LSTM(rng, self.ctx_emb.output, ctx_emb_units, tgt_emb_units, input_mask=T.transpose(context_mask), batch_size=bs) # output: batch_size x tgt_emb_size
        #self.tgt_emb = nn.Embedding(rng, targets, lemma_inv_size, tgt_emb_units, batch_size=bs) # output: batch_size x tgt_emb_size
        #self.query = nn.Combine([self.lstm.output, self.tgt_emb.output], op='mean') # output: batch_size x tgt_emb_size 
        #self.top = nn.Attention(self.query.output, senses, info_mask=senses_mask) # output: batch_size x kge_emb_size 
        
        self.top = nn.Attention(self.lstm.output, senses, info_mask=senses_mask) # output: batch_size x kge_emb_size 

        # PARAMETERS TO BE LEARNT
        #self.params = self.ctx_emb.params + self.lstm.params + self.tgt_emb.params
        #self.deltas = self.ctx_emb.deltas + self.lstm.deltas + self.tgt_emb.deltas
        
        self.params = self.ctx_emb.params + self.lstm.params
        self.deltas = self.ctx_emb.deltas + self.lstm.deltas
        
        # LOSS FUNCTION, COSINE SIMILARITY
        def sentence_cosine(indexes, output):
            output = output + T.exp(-30)
            zero = T.constant(0, dtype='float64')
            cosine = theano.ifelse.ifelse(T.eq(indexes[0], indexes[1]), 
                                       zero,
                                       functions.multi_cosine(output[indexes[0]:indexes[1]]))
            return cosine
               
        cosines, _ = theano.scan(fn=sentence_cosine,
                              outputs_info=None,
                              sequences=[case_indexes],
                              non_sequences=self.top.output)

        self.to_print1 = self.top.relevance
        
        self.loss = -T.sum(cosines)

        # BACK-PROPAGATION, GET GRADIENTS
        self.paramsGrad = [T.grad(self.loss, param) for param in self.params]

        # UPDATE WEIGHTS AND BIASES
        self.optimizer = optimizers.SGD(self.deltas, self.params, self.paramsGrad, batch_size=bs)
        self.updates = self.optimizer.updates()
