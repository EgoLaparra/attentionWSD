#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:54:45 2016

@author: egoitz
"""
import sys
import pickle

import numpy as np
from numpy import random
import theano
from theano import tensor as T

import data
#import theano_lstm as M
import theano_cbow as M
        
dataset_file = sys.argv[2]
epochs=1
sequence_size = 4
batch_size = 100 # s
rng = random.RandomState(12345)        
        
if  (sys.argv[1] == "t"):
    # LOAD DATA
    print ('Loading data...')
    #(vocab, train_set, test_set, valid_set) = data.load_file(dataset_file, sequence_size)
    (vocab, train_set, test_set, valid_set) = data.load_matlab(dataset_file)
    #(vocab, train_set, test_set, valid_set) = data.load_file_full_seq(dataset_file, sequence_size)
    #train_set = train_set[:len(train_set)/4]
    #test_set = test_set[:len(test_set)/4]
    #valid_set = valid_set[:len(valid_set)/4]
    vocab_size = len(vocab) # v
    (train_input, train_target,
     test_input, test_target,
     valid_input, valid_target) = data.load_data(train_set, test_set, valid_set, batch_size)
    (batches, batch_size, words_num) = np.shape(train_input) # b, s, w
    
    x = T.matrix('x',dtype='int64')
    y = T.vector('y',dtype='int64')
    model = M.Model(rng, vocab_size, x=x, y=y)
    
    # COMPILE NET
    print ('Compiling...')
    train = theano.function(
                        inputs=[x,y],
                        outputs=[model.top.output,model.ce],
                        updates=model.updates
                        )

    test = theano.function(
                        inputs=[x,y],
                        outputs=[model.top.output,model.ce]
                        )
    
    
    # TRAIN.
    show_training_ce_after = 100
    show_validation_ce_after = 1000
    count = 0
    for e in range(0,epochs):
        print ('Epoch %d' % (e + 1))
        batch_ce = 0
        train_ce = 0
        for m in range(0, batches):        
            batch_input = train_input[m] # S x W
            batch_target = train_target[m] # S
            (batch_size, words_num) = np.shape(batch_input)
                
            os, ce = train(batch_input, batch_target)
            ce = ce / batch_size
            count =  count + 1;
            batch_ce = batch_ce + (ce - batch_ce) / count
            train_ce = train_ce + (ce - train_ce) / (m + 1)
            sys.stdout.write('\rBatch %d Train CE %.3f - Average CE %.3f' % (m + 1, batch_ce, train_ce))
            sys.stdout.flush()
            if (m + 1) % show_training_ce_after == 0:
                print ('')
                count = 0
                batch_ce = 0
                            
            # VALIDATE
            if (m + 1) % show_validation_ce_after == 0:
                print ('Running validation ...')
                valid_size = np.shape(valid_input)[0]
                os, valid_ce = test(valid_input, valid_target)
                valid_ce = valid_ce / valid_size
                print (' Validation CE %.3f' % valid_ce)
    
    print ('')
    print ('Finished Training.')
    print ('Final Training CE %.3f\n' % train_ce);
    
    # EVALUATE ON VALIDATION SET
    print ('Running final validation ...')
    valid_size = np.shape(valid_input)[0]
    os, valid_ce = test(valid_input, valid_target)
    valid_ce = valid_ce / valid_size
    print (' Final validation CE %.3f' % valid_ce)
    
    # EVALUATE ON TEST SET
    print ('Running final test ...')
    test_size = np.shape(test_input)[0]
    os, test_ce = test(test_input, test_target)
    test_ce = test_ce / test_size
    print (' Final test CE %.3f' % test_ce)
    
    # SAVE THE MODEL
    print ('')
    print ('Saving the model ...')
    f = open('model.pkl', 'wb')
    pickle.dump(vocab, f)
    for param in model.params:
        pickle.dump(param.get_value(), f)
    f.close()
elif (sys.argv[1] == "p"):
    # LOAD THE MODEL
    print ('Loading the model...')
    f = open('model.pkl', 'rb')
    vocab = pickle.load(f)
    vocab_size=len(vocab)
    x = T.matrix('x',dtype='int64')
    model = M.Model(rng, vocab_size, x=x)    
    for param in model.params:
        param.set_value(pickle.load(f))    
    f.close()    
    
    # COMPILE NET
    print ('Compiling...')
    predict = theano.function(inputs=[x], outputs=model.top.output)

    raw_sentence=raw_input('> ')
    while raw_sentence != "":
        sentence = np.array([[vocab.index(word) for word in raw_sentence.split()]])
        prediction = predict(sentence)        
        for idx in np.argsort(prediction[0])[::-1][:5]:
            print (vocab[idx])
        raw_sentence=raw_input('> ')
elif (sys.argv[1] == "s"):
    # LOAD THE MODEL
    print ('Loading the model...')
    f = open('model.pkl', 'rb')
    vocab = pickle.load(f)
    vocab_size=len(vocab)
    x = T.matrix('x',dtype='int64')
    model = M.Model(rng, vocab_size, x=x)    
    for param in model.params:
        param.set_value(pickle.load(f))
    f.close()    
    
    # LOAD EMBEDDINGS
    print ('Loading embeddings...')
    embeddings = model.emb.W.get_value()
    
    raw_word=raw_input('> ')
    while raw_word != "":
        word_idx = vocab.index(raw_word)
        word_emb = embeddings[word_idx]
        numerator = np.dot(embeddings,word_emb)
        denominator = np.sqrt(np.sum(word_emb**2)*np.sum(embeddings**2,axis=1))
        cosine = numerator/denominator
        for idx in np.argsort(cosine)[::-1][:5]:
            print (vocab[idx])
        raw_word=raw_input('> ')