#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:41:13 2016

@author: egoitz
"""

class Optimizer():
    '''
    Optimizer super-class
    '''
    def __init__(self):
        self.deltas = []
        self.params = []
        self.paramsDeltas = []
        self.paramsWeights = []
        
    def updates(self):
        return ([(param, paramWeights) for param, paramWeights in zip(self.params, self.paramsWeights)] 
                 + [(delta, paramDelta) for delta, paramDelta in zip(self.deltas, self.paramsDeltas)])
    
class SGDHinton(Optimizer):
    '''
    Stochastic Gradient Descent. Version by Hinton.
    '''
    def __init__(self, deltas, params, grads, momentum=.9, learning_rate=.1, batch_size=1):
        self.deltas = deltas
        self.params = params
        self.paramsDeltas = [momentum * delta + grad / batch_size
                             for delta, grad in zip(deltas, grads)]
        self.paramsWeights = [param - learning_rate * paramDeltas
                              for param, paramDeltas in zip(params, self.paramsDeltas)]
        
class SGD(Optimizer):
    '''
    Stochastic Gradient Descent
    '''
    def __init__(self, deltas, params, grads, momentum=.9, learning_rate=.1, batch_size=1):
        self.deltas = deltas
        self.params = params
        self.paramsDeltas = [momentum * delta - learning_rate * grad / batch_size
                             for delta, grad in zip(deltas, grads)]
        self.paramsWeights = [param + paramDeltas
                              for param, paramDeltas in zip(params, self.paramsDeltas)]
