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
from theano.compile.nanguardmode import NanGuardMode

import data
import net as M 
       
epochs=1
batch_size = 100
rng = random.RandomState(12345)        
        
if  (sys.argv[1] == "t"):
    # LOAD DATA
    print ('Loading data...')
    wnkge, wndict = data.load_kge()            
    vocab, lemma_inv, max_context, max_polisemy, dataset = data.load_penn_treebank(wnkge, wndict)
    vocab_size = len(vocab)
    lemma_inv_size = len(lemma_inv)    
    train_set, valid_set, test_set = data.load_sets(dataset)
    (train_targets, train_contexts, train_context_mask,
     train_senses, train_senses_mask, train_indexes) = data.load_data(train_set, max_context,
                                                                        max_polisemy, batch_size=batch_size)
    batches = len(train_targets)
    (valid_targets, valid_contexts, valid_context_mask,
     valid_senses, valid_senses_mask, valid_indexes) = data.load_data(valid_set, max_context, max_polisemy)
    (test_targets, test_contexts, test_context_mask,
     test_senses, test_senses_mask, test_indexes) = data.load_data(test_set, max_context, max_polisemy)
     
    x_targets = T.vector('x_targets',dtype='int64') # batch_size
    x_contexts = T.matrix('x_contexts',dtype='int64')  # batch_size X num_words
    x_context_mask = T.matrix('x_context_mask',dtype='int64')  # batch_size X num_words
    x_senses = T.tensor3('x_senses',dtype='float64')  # batch_size X num_senses X kge_size
    x_senses_mask = T.matrix('x_senses_mask',dtype='int64')  # batch_size X num_senses x kge_size
    x_indexes = T.matrix('x_indexes',dtype='int64')  # num_sentences X 2
    model = M.Model(rng, vocab_size, lemma_inv_size, batch_size,
                    targets=x_targets, contexts=x_contexts, context_mask=x_context_mask,
                    senses=x_senses, senses_mask=x_senses_mask, case_indexes=x_indexes)
    
    # COMPILE NET
    print ('Compiling...')
#    train = theano.function(
#                        inputs=[x_targets, x_contexts, x_context_mask, 
#                                x_senses, x_senses_mask, x_indexes],
#                        outputs=[model.top.output,model.loss],
#                        updates=model.updates
#                        )
    
    trainfake = theano.function(
                        inputs=[x_targets, x_contexts, x_context_mask, 
                                x_senses, x_senses_mask, x_indexes],
                        outputs=[model.top.output,model.loss],
                        updates=model.updates,
                        )

    test = theano.function(
                        inputs=[x_targets, x_contexts, x_context_mask, 
                                x_senses, x_senses_mask, x_indexes],
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
            batch_context_mask = np.transpose(train_context_mask[b])
            batch_senses = train_senses[b]
            batch_senses_mask = train_senses_mask[b]
            batch_indexes = train_indexes[b]

            out, loss = trainfake(batch_targets, batch_contexts, batch_context_mask,
                              batch_senses, batch_senses_mask, batch_indexes)
            
            
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
                valid_size = np.size(valid_targets)
                out, valid_loss = test(valid_targets, valid_contexts, valid_context_mask,
                                       valid_senses, valid_senses_mask, valid_indexes)
                valid_loss = valid_loss / valid_size
                print (' Validation COSINE %.3f' % valid_loss)
    
    print ('')
    print ('Finished Training.')
    print ('Final Training COSINE %.3f\n' % train_loss);
    
    # EVALUATE ON VALIDATION SET
    print ('Running final validation ...')
    valid_size = np.size(valid_targets)
    out, valid_loss = test(valid_targets, valid_contexts, valid_context_mask,
                           valid_senses, valid_senses_mask, valid_indexes)
    valid_loss = valid_loss / valid_size
    print (' Final validation CE %.3f' % valid_loss)
    
    # EVALUATE ON TEST SET
    print ('Running final test ...')
    test_size = np.size(test_targets)
    out, test_loss = test(test_targets, test_contexts, test_context_mask,
                          test_senses, test_senses_mask, test_indexes)
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
