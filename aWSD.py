#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:54:45 2016

@author: egoitz
"""
import sys
sys.path.append('/home/egoitz/Tools/embeddings/kge/SME/')
import pickle

import numpy as np
from numpy import random
import theano
from theano import tensor as T

import data
#import theano_lstm as M
#import theano_cbow as M
import net as M 
       
epochs=1
batch_size = 100 # s
rng = random.RandomState(12345)        
        
if  (sys.argv[1] == "t"):
    # LOAD DATA
    print ('Loading data...')
    wnkge, wndict = data.load_kge()            
    vocab, lemma_inv, dataset = data.load_penn_treebank(wnkge, wndict)
    vocab_size = len(vocab)
    lemma_inv_size = len(lemma_inv)    
    train_set, valid_set, test_set = data.load_sets(dataset)
    (train_targets, train_contexts,
     train_senses, train_indexes) = data.load_data(train_set, batch_size)
    batches = len(train_targets)
    (valid_targets, valid_contexts,
     valid_senses, valid_indexes) = data.load_data(valid_set)
    (test_targets, test_contexts,
     test_senses, test_indexes) = data.load_data(test_set)

    x_targets = T.matrix('x',dtype='int64')
    x_contexts = T.matrix('x',dtype='int64')
    x_senses = T.matrix('x',dtype='float64')
    x_indexes = T.matrix('x',dtype='int64')
    model = M.Model(rng, vocab_size, lemma_inv_size, batch_size,
                    targets=x_targets, contexts=x_contexts,
                    senses=x_senses, case_indexes=x_indexes)
    
    # COMPILE NET
    print ('Compiling...')
    train = theano.function(
                        inputs=[x_targets, x_contexts, x_senses, x_indexes],
                        outputs=[model.top.output,model.loss],
                        updates=model.updates
                        )

    test = theano.function(
                        inputs=[x_targets, x_contexts, x_senses, x_indexes],
                        outputs=[model.top.output,model.loss]
                        )
    
    
    # TRAIN.
    show_training_loss_after = 100
    show_validation_loss_after = 1000
    count = 0
    for e in range(0,epochs):
        print ('Epoch %d' % (e + 1))
        batch_loss = 0
        train_loss = 0
        for b in range(0, batches):        
            batch_targets = train_targets[b]
            batch_contexts = train_contexts[b]
            batch_senses = train_senses[b]
            batch_indexes = train_indexes[b]

            #(batch_size, words_num) = np.shape(batch_input)
                
            out, loss = train(batch_targets, batch_contexts, batch_senses, batch_indexes)
            loss = loss / batch_size
            count =  count + 1;
            batch_loss = batch_loss + (loss - batch_loss) / count
            train_loss = train_loss + (loss - train_loss) / (b + 1)
            sys.stdout.write('\rBatch %d Train COSINE %.3f - Average COSINE %.3f' % (b + 1, batch_loss, train_loss))
            sys.stdout.flush()
            if (b + 1) % show_training_loss_after == 0:
                print ('')
                count = 0
                batch_ce = 0
                            
            # VALIDATE
            if (b + 1) % show_validation_loss_after == 0:
                print ('Running validation ...')
                valid_size = np.shape(valid_targets)[1]
                out, valid_loss = test(valid_targets, valid_contexts, valid_senses, valid_indexes)
                valid_loss = valid_loss / valid_size
                print (' Validation COSINE %.3f' % valid_loss)
    
    print ('')
    print ('Finished Training.')
    print ('Final Training COSINE %.3f\n' % train_loss);
    
    # EVALUATE ON VALIDATION SET
    print ('Running final validation ...')
    valid_size = np.shape(valid_targets)[1]
    out, valid_loss = test(valid_targets, valid_contexts, valid_senses, valid_indexes)
    valid_loss = valid_loss / valid_size
    print (' Final validation CE %.3f' % valid_loss)
    
    # EVALUATE ON TEST SET
    print ('Running final test ...')
    test_size = np.shape(test_targets)[1]
    out, test_loss = test(test_targets, test_contexts, test_senses, test_indexes)
    test_loss = test_loss / test_size
    print (' Final test CE %.3f' % test_loss)
    
    # SAVE THE MODEL
    print ('')
    print ('Saving the model ...')
    f = open('model.pkl', 'wb')
    pickle.dump(vocab, f)
    pickle.dump(lemma_inv, f)
    for param in model.params:
        pickle.dump(param.get_value(), f)
    f.close()
