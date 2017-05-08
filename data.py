#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:46:06 2016

@author: egoitz
"""
import sys
import re
import pickle

import numpy as np
from numpy.random import shuffle

from nltk.corpus.reader import BracketParseCorpusReader as bpcr
from nltk.corpus import treebank
from nltk.stem import WordNetLemmatizer
wn_lemmatizer = WordNetLemmatizer()

#def load_file (file_name, N):
#    '''
#    Load the dataset file. Create vocabulary and train, test and valid sets. 
#    Each example in set is a sequence of N tokens.
#    
#    Inputs:
#        file_name: Dataset file. One sentence per line. Tokens separated by
#                    spaces.
#        N: number of tokens per example sequence
#    Outputs:
#        vocab: 
#        train_set:
#        test_set:
#        valid_set:
#    '''
#    vocab = []
#    test_size_perc = .1
#    valid_size_perc = .1
#    data_set = []
#    datafile = open(file_name, 'r')
#    for sent in datafile:
#        tokens = sent.strip().split()
#        for t in range(0,len(tokens) - (N - 1)):
#            sent = []
#            for n in range(0, N):
#                lc_token = tokens[t+n].lower()
#                if lc_token not in vocab:
#                    vocab.append(lc_token)
#                tokid = vocab.index(lc_token)
#                sent.append(tokid)
#            data_set.append(sent)
#    datafile.close()
#    
#    shuffle(data_set)
#    test_last = int(len(data_set) * test_size_perc)
#    valid_last = int(test_last + len(data_set) * valid_size_perc)
#    test_set = data_set[:test_last]
#    valid_set = data_set[test_last:valid_last]
#    train_set = data_set[valid_last:]
#    
#    return vocab, train_set, test_set, valid_set


#def load_data(train_set, test_set, valid_set, M):
#    '''
#    Create train, test and valid inputs and targets. Divide train in Mini-batches.
#    
#    Inputs:
#        train_set:
#        test_set:
#        valid_set:
#        M: Mini-batch size
#        
#    Outputs:
#        :
#
#    '''
#    num_batches = int(len(train_set) / M)
#    train_input = [t[:-1] for t in train_set]
#    train_input = np.resize(train_input,(num_batches,M,np.shape(train_input)[-1]))
#    train_target = [t[-1] for t in train_set]
#    train_target = np.resize(train_target,(num_batches,M))
#    test_input = [t[:-1] for t in test_set]
#    test_target = [t[-1] for t in test_set]
#    valid_input = [v[:-1] for v in valid_set]
#    valid_target = [v[-1] for v in valid_set]
# 
#    return train_input, train_target, test_input, test_target, valid_input, valid_target
    


def load_data(set_data, max_context, max_polisemy, batch_size=None):
    '''
    Create target, context and sense sets
    '''
    set_targets_idx = []
    set_targets = []
    set_contexts = []
    set_context_mask = []
    set_senses = []
    set_senses_mask = []
    set_indexes = []
    batch_targets_idx = []
    batch_targets = []
    batch_contexts = []
    batch_context_mask = []
    batch_senses = []
    batch_senses_mask = []
    batch_indexes = []
    case_index = -1
    for sent in set_data:
        sent_indexes = []
        for t in range(0, len(sent[0])):
            if batch_size == None:
                set_targets_idx.append(sent[0][t])
                set_targets.append(sent[1][t])
            else:
                batch_targets_idx.append(sent[0][t])
                batch_targets.append(sent[1][t])
            context = list(sent[2][t])
            context_mask = list(np.ones(len(context), dtype='int64'))
            for c in range(len(context), max_context):
                context.append(1)
                context_mask.append(0)
            if batch_size == None:
                set_contexts.append(context)
                set_context_mask.append(context_mask)
            else:
                batch_contexts.append(context)
                batch_context_mask.append(context_mask)
            senses = list(sent[3][t])
            senses_mask = list(np.ones(len(senses), dtype='int64'))
            for s in range(len(senses), max_polisemy):
                senses.append(list(np.ones(20, dtype='int64')))
                senses_mask.append(0)
            if batch_size == None:
                set_senses.append(senses)
                set_senses_mask.append(senses_mask)
            else:
                batch_senses.append(senses)
                batch_senses_mask.append(senses_mask)
                case_index += 1
            sent_indexes.append(case_index)            
            if batch_size != None and len(batch_targets) == batch_size:
                idx = []
                idx.append(sent_indexes[0])
                idx.append(sent_indexes[-1] + 1)
                batch_indexes.append(idx)
                sent_indexes = []
                for i in range(len(batch_indexes), 20):
                    batch_indexes.append([0,0])
                set_targets_idx.append(batch_targets_idx)
                set_targets.append(batch_targets)
                set_contexts.append(batch_contexts)
                set_context_mask.append(batch_context_mask)
                set_senses.append(batch_senses)
                set_senses_mask.append(batch_senses_mask)
                set_indexes.append(batch_indexes)
                batch_targets_idx = []
                batch_targets = []
                batch_contexts = []
                batch_context_mask = []
                batch_senses = []
                batch_senses_mask = []
                batch_indexes = []
                case_index = -1
        if len(sent_indexes) > 0 :
            if batch_size == None:
                idx = []
                idx.append(sent_indexes[0])
                idx.append(sent_indexes[-1] + 1)
                set_indexes.append(idx)
            else:
                idx = []
                idx.append(sent_indexes[0])
                idx.append(sent_indexes[-1] + 1)
                batch_indexes.append(idx)


    return (set_targets_idx, set_targets, set_contexts, set_context_mask,
            set_senses, set_senses_mask, set_indexes)
        

def load_sets(dataset, test_size_perc = .1, valid_size_perc = .1):
    '''
    Create train, test and valid sets from dataset
    '''
    #shuffle(dataset)
    test_last = int(len(dataset) * test_size_perc)
    valid_last = int(test_last + len(dataset) * valid_size_perc)
    test_set = dataset[:test_last]
    valid_set = dataset[test_last:valid_last]
    train_set = dataset[valid_last:]

    #train_set = [list(train_set[0]) for i in range(0,2000)]
                     
    return train_set, valid_set, test_set
    
   
def load_corpus(config, wnkge, wndict, corpus='treebank', vocab=[], lemma_inv=[]):
    '''
    Load Penn Treebank Corpus
    '''
    
    if corpus == 'treebank':
        corpus = treebank
    elif corpus == 's3aw':
        doclist = []
        for doc in config.get('Senseval','files').split(','):
            doclist.append(doc)
        corpus = bpcr(config.get('Senseval','path'), doclist)
    
    new_vocab = True
    if np.size(vocab) > 0:
        new_vocab = False
    if new_vocab:
        vocab.append('$UNK$')
        vocab.append('$TARGET$')
    new_lemma_inv = True
    if np.size(lemma_inv) > 0:
        new_lemma_inv = False
    max_context = 0
    max_polisemy = 0
    dataset = []
    d = -1
    for fileid in corpus.fileids():
        d += 1
        s = -1
        for sent in corpus.tagged_sents(fileid):
            s += 1
            t = -1
            sentence = []
            words = []
            target_words = []
            target_lemmas = []
            target_ids = []
            for w in range(0,len(sent)):
                t += 1
                token = sent[w]
                word = token[0]
                if token[1] != '-NONE-':
                    if word not in vocab:
                        if not new_vocab:
                            word = '$UNK$'
                        else:
                            vocab.append(word)
                    words.append(vocab.index(word))
                if re.search(r'(^NN[^P]*$|^VB|^JJ|^RB)',token[1]):
                    pos = re.sub(r'^NN.*','n',token[1])
                    pos = re.sub(r'^VB.*','v',pos)
                    pos = re.sub(r'^JJ.*','a',pos)
                    pos = re.sub(r'^RB.*','r',pos)
                    lemma = wn_lemmatizer.lemmatize(token[0],pos=pos) + '-' + pos
                    if lemma in wndict and (lemma in lemma_inv or new_lemma_inv): # The lemma is in wordnet and in the lemma inventory ("test") or we are creating the lemma inventory ("trainig")
                        tid = "d%03i.s%03i.t%03i" % (d,s,t)
                        target_ids.append(tid)
                        target_words.append(w)
                        if lemma not in lemma_inv:
                            lemma_inv.append(lemma)
                        target_lemmas.append(lemma)
                        
            ids = []
            targets=[]
            contexts=[]
            senses=[]
            for tw in range(0,len(target_words)):
                t = target_words[tw]
                context = []
                for w in range(0,len(words)):
                    if w == t:
                        context.append(vocab.index('$TARGET$'))
                    else:
                        context.append(words[w])
                word_senses = []
                for sense in wndict[target_lemmas[tw]]:
                    if sense in wnkge:
                        word_senses.append(wnkge[sense])
                ids.append(target_ids[tw])
                targets.append(lemma_inv.index(target_lemmas[tw]))
                contexts.append(context)
                senses.append(word_senses)
                if len(context) > max_context:
                    max_context = len(context)
                if len(word_senses) > max_polisemy:
                    max_polisemy = len(word_senses)
            sentence.append(ids)
            sentence.append(targets)
            sentence.append(contexts)
            sentence.append(senses)
            dataset.append(sentence)
    
    return vocab, lemma_inv, max_context, max_polisemy, dataset   
                
#def load_kge ():
#    '''
#    Load Knowledge Graph Embeddings
#    '''
#    f = open('/home/egoitz/Tools/embeddings/kge/SME/WN/data/WN_synset2idx.pkl', 'rb')
#    s2i = pickle.load(f)
#    f = open('/home/egoitz/Tools/embeddings/kge/SME/WN/data/WN_synset2concept.pkl', 'rb')
#    s2c = pickle.load(f)
#    f = open('/home/egoitz/Tools/embeddings/kge/SME/WN/WN_TransE/best_valid_model.pkl', 'rb') 
#    m= pickle.load(f)
#    kge = zip(*m[0].E.get_value())
#    
#    wn_dict = dict()
#    for s in s2c:
#        c = re.sub(r'_[0-9]+$','',re.sub(r'^__','',s2c[s]))
#        c = re.sub(r'_NN.*','_n',c)
#        c = re.sub(r'_VB.*','_v',c)
#        c = re.sub(r'_JJ.*','_a',c)
#        c = re.sub(r'_RB.*','_r',c)
#        if c not in wn_dict:
#            wn_dict[c] = list()
#        wn_dict[c].append(s)
#    
#    wn_kge = dict()
#    for s in s2i:
#        i = s2i[s]
#        wn_kge[s] = kge[i]
#        
#    return wn_kge, wn_dict

def load_kge (config):
    '''
    Load Knowledge Graph Embeddings
    '''       
    f = open(config.get('KGE','s2i'), 'rb')
    s2i = pickle.load(f)
    f = open(config.get('KGE','model'), 'rb') 
    m= pickle.load(f)
    kge = zip(*m[0].E.get_value())    
    wn_kge = dict()
    for s in s2i:
        i = s2i[s]
        wn_kge[s] = kge[i]
    f.close()
    
    s2c = open(config.get('KGE','senses'), 'r')
    wn_dict = dict()
    for line in s2c:
        fields = line.rstrip().split()
        c = fields[0]
        slist = list()
        for s in fields[1:]:
            if s in wn_kge:
                slist.append(s)
        if len(slist) > 0:
            wn_dict[c] = slist
    s2c.close()

    return wn_kge, wn_dict
