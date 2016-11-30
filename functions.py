#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:09:39 2016

@author: egoitz
"""
import theano.tensor as T

def sigmoid (x):
    return 1 / (1 + T.exp(-x))
    
def softmax (x):
    e = T.exp(x)
    return e / T.reshape(T.sum(e, axis=1), (T.shape(x)[0],1))
    
def categorical_crossentropy(x,y):
    return -T.sum(y * T.log(x))
    
def sum_cross_entropy(x,y):
    t = T.exp(-30) # Avoids divide by zero in log        
    return T.sum(categorical_crossentropy((x + t), y))
    
def negative_sampling(rng, x, y):    
    K = 5   # number of negative samples required.
    random_numbers = rng.random_integers(size=(K), low=0, high=3)
    sample = x[:,random_numbers]
    return -T.sum(y * (T.log(sigmoid(x)) + T.reshape(T.sum(T.log(sigmoid(-sample)), axis=1), (T.shape(x)[0],1))))
    
def multi_cosine(x):
    #numerator = T.dot(x,T.transpose(x))
    square_sum = T.sum(x**2,axis=1)
    denominator = T.sqrt(square_sum * T.transpose(T.tile(square_sum, (T.shape(x)[0],1))))                                                                                    
    #cosine = T.tril(numerator/denominator,k=-1)
    cosine = denominator
    return T.sum(cosine)
    
def cosine(x,y):
    numerator = T.sum(y * x, axis=1)
    denominator = T.sqrt(T.sum(y**2, axis=1) * T.sum(x**2, axis=1))
    cosine = numerator/denominator
    return  cosine