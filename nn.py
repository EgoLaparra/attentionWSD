#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:03:13 2016

@author: egoitz
"""

import numpy as np
import theano
import theano.tensor as T

class Combine:
    '''
    Combine tensors Layer
    '''
    
    def __init__(self, input, axis=0, op='sum'):

        if op == 'sum':
            self.output = T.sum(input, axis=axis)
        elif op == 'mean':
            self.output = T.mean(input, axis=axis)
        elif op == 'prod':
            self.output = T.prod(input, axis=axis)
        elif op == 'max':
            self.output = T.max(input, axis=axis)
        elif op == 'min':
            self.output = T.min(input, axis=axis)
        elif op == 'conc':
            self.output = T.concatenate(input, axis=axis)

            
class Embedding:
    """ 
    Embedding Layer
    """
    
    def __init__(self, rng, input, n_in, n_out, 
                 W=None, WD=None, 
                 batch_size=1, sequence=False): # EmbeddingLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        if W is None:
            W_values = np.asarray (
                                   rng.normal(scale = .01, size = (n_in, n_out)),
                                   dtype = theano.config.floatX
                                   )
            W = theano.shared(value=W_values, name="W", borrow=True)
        self.W = W
            
        self.params = [self.W]
        
        ## Initialize deltas
        if WD is None:
            WD_values = np.zeros((n_in, n_out))
            WD = theano.shared(WD_values, name='W_D') 
        self.WD = WD
            
        self.deltas = [WD]

        ## Compute activation function
        if sequence:
            self.output = self.W[input,:].dimshuffle(1,0,2)
        else:
            self.output = T.reshape(self.W[input,:],(batch_size,-1))
        
        
        ## Keep track of input
        self.input = input

        
class Dense:
    """ 
    Dense (Fully-Connected) Layer
    """
    
    def __init__(self, rng, input, n_in, n_out, 
                 W=None, b=None, WD=None, bD=None, 
                 batch_size=1, activation=T.nnet.sigmoid): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        if W is None:
            W_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            W = theano.shared(value=W_values, name="W", borrow=True)
        self.W = W
            
        ## Initialize the biases with 0s as a n_out vector
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)
        self.b = b

        self.params = [self.W, self.b]

        ## Initialize deltas
        if WD is None:
            WD_values = np.zeros((n_in, n_out))
            WD = theano.shared(WD_values, name='W_D') 
        self.WD = WD
        if bD is None:
            bD_values = np.zeros(n_out)
            bD = theano.shared(bD_values, name='b_D') 
        self.bD = bD
            
        self.deltas = [WD, bD]

        ## Compute activation function
        linear_output = T.dot(input, self.W) + self.b
        if activation == None:
            self.output = linear_output
        else:
            self.output = activation(linear_output)
        
        self.input = input


        
class RNN:
    '''
    Recurrent layer
    '''
    def __init__(self, rng, input, n_in, n_out, 
                 W=None, Wt=None, b=None, o0=None, 
                 WD=None, WtD=None, bD=None, o0D=None,
                 batch_size=1, activation_hidden=T.nnet.sigmoid,
                 sequence=False, inverse=False): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        if W is None:
            W_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            W = theano.shared(value=W_values, name="W", borrow=True)
        self.W = W

        if Wt is None:
            Wt_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wt = theano.shared(value=Wt_values, name="Wt", borrow=True)
        self.Wt = Wt
                        
        ## Initialize the biases with 0s as a n_out vector
        if b is None:
            b_values = np.zeros((n_in,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)
        self.b = b

        ## Initialize hidden state for t = 0
        if o0 is None:
            o0_values = np.zeros((n_out,), dtype=theano.config.floatX)
            o0 = theano.shared(value=o0_values, name="o0", borrow=True)
        self.o0 = o0

        self.params = [self.W, self.Wt, self.b, self.o0]

        ## Initialize deltas
        if WD is None:
            WD_values = np.zeros((n_in, n_out))
            WD = theano.shared(WD_values, name='W_D') 
        self.WD = WD
        if WtD is None:
            WtD_values = np.zeros((n_out, n_out))
            WtD = theano.shared(WtD_values, name='Wt_D') 
        self.WtD = WtD
        if bD is None:
            bD_values = np.zeros(n_in)
            bD = theano.shared(bD_values, name='b_D') 
        self.bD = bD
        if o0D is None:
            o0D_values = np.zeros((n_out,), dtype=theano.config.floatX)
            o0D = theano.shared(value=o0D_values, name="o0D", borrow=True)
        self.o0D = o0D
           
        self.deltas = [WD, WtD, bD, o0D]

        ## Compute recurrence
        def recurrence(input_t, output_tm1):
            linear_output = T.dot(input_t, self.W) + T.dot(output_tm1, self.Wt) + self.b
            output_t = activation_hidden(linear_output)
            return output_t

        output, _ = theano.scan(
                                fn=recurrence,
                                sequences=input[::-1] if inverse else input,
                                outputs_info=T.tile(self.o0,(batch_size, 1)),
                                n_steps=input.shape[0]
                                )
        
        if sequence:
            self.output = output
        else:
            self.output = output[0] if inverse else output[-1]
            
        self.input = input
    
        
class Elman:
    '''
    Elman recurrent layer
    '''
    def __init__(self, rng, input, n_in, n_hidden, n_out, 
                 Wh=None, Wt=None, Wo=None, bh=None, bo=None, h0=None, 
                 WhD=None, WtD=None, WoD=None, bhD=None, boD=None, h0D=None,
                 batch_size=1, activation_hidden=T.nnet.sigmoid, activation_output=T.nnet.softmax,
                 sequence=False): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        if Wh is None:
            Wh_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_hidden)),
                                  dtype = theano.config.floatX
                                  )
            Wh = theano.shared(value=Wh_values, name="Wh", borrow=True)
        self.Wh = Wh

        if Wt is None:
            Wt_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_hidden, n_hidden)),
                                  dtype = theano.config.floatX
                                  )
            Wt = theano.shared(value=Wt_values, name="Wt", borrow=True)
        self.Wt = Wt
        
        if Wo is None:
            Wo_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_hidden, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wo = theano.shared(value=Wo_values, name="Wo", borrow=True)
        self.Wo = Wo
                
        ## Initialize the biases with 0s as a n_out vector
        if bh is None:
            bh_values = np.zeros((n_in,), dtype=theano.config.floatX)
            bh = theano.shared(value=bh_values, name="bh", borrow=True)
        self.bh = bh

        if bo is None:
            bo_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bo = theano.shared(value=bo_values, name="bo", borrow=True)
        self.bo = bo

        ## Initialize hidden state for t = 0
        if h0 is None:
            h0_values = np.zeros((n_hidden,), dtype=theano.config.floatX)
            h0 = theano.shared(value=h0_values, name="h0", borrow=True)
        self.h0 = h0

        self.params = [self.Wh, self.Wt, self.Wo, self.bh, self.bo, self.h0]

        ## Initialize deltas
        if WhD is None:
            WhD_values = np.zeros((n_in, n_hidden))
            WhD = theano.shared(WhD_values, name='Wh_D') 
        self.WhD = WhD
        if WtD is None:
            WtD_values = np.zeros((n_hidden, n_hidden))
            WtD = theano.shared(WtD_values, name='Wt_D') 
        self.WtD = WtD
        if WoD is None:
            WoD_values = np.zeros((n_hidden, n_out))
            WoD = theano.shared(WoD_values, name='Wo_D') 
        self.WoD = WoD
        if bhD is None:
            bhD_values = np.zeros(n_in)
            bhD = theano.shared(bhD_values, name='bh_D') 
        self.bhD = bhD
        if boD is None:
            boD_values = np.zeros(n_out)
            boD = theano.shared(boD_values, name='bo_D') 
        self.boD = boD
        if h0D is None:
            h0D_values = np.zeros((n_hidden,), dtype=theano.config.floatX)
            h0D = theano.shared(value=h0D_values, name="h0D", borrow=True)
        self.h0D = h0D
           
        self.deltas = [WhD, WtD, WoD, bhD, boD, h0D]

        ## Compute recurrence
        def recurrence(input_t, hidden_tm1):
            linear_hidden = T.dot(input_t, self.Wh) + T.dot(hidden_tm1, self.Wt) + self.bh
            hiddent_t = activation_hidden(linear_hidden)
            linear_output = T.dot(hiddent_t, self.Wo) + self.bo
            output_t = activation_output(linear_output)
            return [hiddent_t, output_t]

        [hidden, output], _ = theano.scan(
                                fn=recurrence,
                                sequences=input,
                                outputs_info=[T.tile(self.h0,(batch_size, 1)), None],
                                n_steps=input.shape[0]
                                )
        
        if sequence:
            self.hidden = hidden
            self.output = output
        else:
            self.hidden = hidden[-1]
            self.output = output[-1]
            
        self.input = input

        
class Jordan:
    '''
    Jordan recurrent layer
    '''
    def __init__(self, rng, input, n_in, n_hidden, n_out, 
                 Wh=None, Wt=None, Wo=None, bh=None, bo=None, o0=None, 
                 WhD=None, WtD=None, WoD=None, bhD=None, boD=None, o0D=None,
                 batch_size=1, activation_hidden=T.nnet.sigmoid, activation_output=T.nnet.softmax,
                 sequence=False): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        if Wh is None:
            Wh_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_hidden)),
                                  dtype = theano.config.floatX
                                  )
            Wh = theano.shared(value=Wh_values, name="Wh", borrow=True)
        self.Wh = Wh

        if Wt is None:
            Wt_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_hidden)),
                                  dtype = theano.config.floatX
                                  )
            Wt = theano.shared(value=Wt_values, name="Wt", borrow=True)
        self.Wt = Wt
     
        if Wo is None:
            Wo_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_hidden, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wo = theano.shared(value=Wo_values, name="Wo", borrow=True)
        self.Wo = Wo
                
        ## Initialize the biases with 0s as a n_out vector
        if bh is None:
            bh_values = np.zeros((n_in,), dtype=theano.config.floatX)
            bh = theano.shared(value=bh_values, name="bh", borrow=True)
        self.bh = bh

        if bo is None:
            bo_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bo = theano.shared(value=bo_values, name="bo", borrow=True)
        self.bo = bo

        ## Initialize output state for t = 0
        if o0 is None:
            o0_values = np.zeros((n_out,), dtype=theano.config.floatX)
            o0 = theano.shared(value=o0_values, name="o0", borrow=True)
        self.o0 = o0

        self.params = [self.Wh, self.Wt, self.Wo, self.bh, self.bo, self.o0]

        ## Initialize deltas
        if WhD is None:
            WhD_values = np.zeros((n_in, n_hidden))
            WhD = theano.shared(WhD_values, name='Wh_D') 
        self.WhD = WhD
        if WtD is None:
            WtD_values = np.zeros((n_out, n_hidden))
            WtD = theano.shared(WtD_values, name='Wt_D') 
        self.WtD = WtD
        if WoD is None:
            WoD_values = np.zeros((n_hidden, n_out))
            WoD = theano.shared(WoD_values, name='Wo_D') 
        self.WoD = WoD
        if bhD is None:
            bhD_values = np.zeros(n_in)
            bhD = theano.shared(bhD_values, name='bh_D') 
        self.bhD = bhD
        if boD is None:
            boD_values = np.zeros(n_out)
            boD = theano.shared(boD_values, name='bo_D') 
        self.boD = boD
        if o0D is None:
            o0D_values = np.zeros((n_out,), dtype=theano.config.floatX)
            o0D = theano.shared(value=o0D_values, name="o0D", borrow=True)
        self.o0D = o0D
        
        self.deltas = [WhD, WtD, WoD, bhD, boD, o0D]

        ## Compute recurrence
        def recurrence(input_t, output_tm1):
            linear_hidden = T.dot(input_t, self.Wh) + T.dot(output_tm1, self.Wt) + self.bh
            hiddent_t = activation_hidden(linear_hidden)
            linear_output = T.dot(hiddent_t, self.Wo) + self.bo
            output_t = activation_output(linear_output)
            return [hiddent_t, output_t]

        [hidden, output], _ = theano.scan(
                                fn=recurrence,
                                sequences=input,
                                outputs_info=[None, T.tile(self.o0,(batch_size, 1))],
                                n_steps=input.shape[0]
                                )
        
        if sequence:
            self.hidden = hidden
            self.output = output
        else:
            self.hidden = hidden[-1]
            self.output = output[-1]
            
        self.input = input
        
        
class LSTM:
    '''
    Long-Short Term Memory layer
    '''
    def __init__(self, rng, input, n_in, n_out, 
                 Wi=None, Wf=None, Wc=None, Wo=None,
                 Wti=None, Wtf=None, Wtc=None, Wto=None,
                 Wci=None, Wcf=None, Wco=None, # Peephole weights
                 bi=None, bf=None, bc=None, bo=None, c0=None, h0=None,
                 WiD=None, WfD=None, WcD=None, WoD=None, 
                 WtiD=None, WtfD=None, WtcD=None, WtoD=None,
                 WciD=None, WcfD=None, WcoD=None, # Peephole deltas
                 biD=None, bfD=None, bcD=None, boD=None, c0D=None, h0D=None,
                 batch_size=1, activation=T.tanh, inner_activation=T.nnet.sigmoid,forget_bias_init=np.ones,
                 sequence=False, inverse=False,
                 peephole_input=False, peephole_forget=False, peephole_output=False): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        ## and biases with 0s as a n_out vector
        if Wi is None:
            Wi_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wi = theano.shared(value=Wi_values, name="Wi", borrow=True)
        self.Wi = Wi

        if Wti is None:
            Wti_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wti = theano.shared(value=Wti_values, name="Wti", borrow=True)
        self.Wti = Wti

        if bi is None:
            bi_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bi = theano.shared(value=bi_values, name="bi", borrow=True)
        self.bi = bi

        if Wf is None:
            Wf_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wf = theano.shared(value=Wf_values, name="Wf", borrow=True)
        self.Wf = Wf

        if Wtf is None:
            Wtf_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wtf = theano.shared(value=Wtf_values, name="Wtf", borrow=True)
        self.Wtf = Wtf

        if bf is None:
            bf_values = forget_bias_init((n_out,), dtype=theano.config.floatX)
            bf = theano.shared(value=bf_values, name="bf", borrow=True)
        self.bf = bf

        if Wc is None:
            Wc_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wc = theano.shared(value=Wc_values, name="Wc", borrow=True)
        self.Wc = Wc

        if Wtc is None:
            Wtc_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wtc = theano.shared(value=Wtc_values, name="Wtc", borrow=True)
        self.Wtc = Wtc

        if bc is None:
            bc_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bc = theano.shared(value=bc_values, name="bc", borrow=True)
        self.bc = bc
        
        if Wo is None:
            Wo_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wo = theano.shared(value=Wo_values, name="Wo", borrow=True)
        self.Wo = Wo
        
        if Wto is None:
            Wto_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wto = theano.shared(value=Wto_values, name="Wto", borrow=True)
        self.Wto = Wto

        if bo is None:
            bo_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bo = theano.shared(value=bo_values, name="bo", borrow=True)
        self.bo = bo

        ## Initialize hidden and cell state for t = 0
        if c0 is None:
            c0_values = np.zeros((n_out,), dtype=theano.config.floatX)
            c0 = theano.shared(value=c0_values, name="c0", borrow=True)
        self.c0 = c0
        
        if h0 is None:
            h0_values = np.zeros((n_out,), dtype=theano.config.floatX)
            h0 = theano.shared(value=h0_values, name="h0", borrow=True)
        self.h0 = h0

        self.params = [self.Wi, self.Wf, self.Wc, self.Wo, 
                       self.Wti, self.Wtf, self.Wtc, self.Wto,
                       self.bi, self.bf, self.bc, self.bo, self.c0, self.h0]

        ## Initialize the peephole Weights

        if peephole_input:
            if Wci is None:
                Wci_values = np.asarray(
                                      rng.normal(scale = .01, size = (n_out, n_out)),
                                      dtype = theano.config.floatX
                                      )
                Wci = theano.shared(value=Wci_values, name="Wci", borrow=True)
            self.Wci = Wci
            self.params.append(self.Wci)
            
        if peephole_forget:
            if Wcf is None:
                Wcf_values = np.asarray(
                                      rng.normal(scale = .01, size = (n_out, n_out)),
                                      dtype = theano.config.floatX
                                      )
                Wcf = theano.shared(value=Wcf_values, name="Wcf", borrow=True)
            self.Wcf = Wcf
            self.params.append(self.Wcf)
            
        if peephole_output:
            if Wco is None:
                Wco_values = np.asarray(
                                      rng.normal(scale = .01, size = (n_out, n_out)),
                                      dtype = theano.config.floatX
                                      )
                Wco = theano.shared(value=Wco_values, name="Wco", borrow=True)
            self.Wco = Wco
            self.params.append(self.Wco)
                
        ## Initialize deltas
        if WiD is None:
            WiD_values = np.zeros((n_in, n_out))
            WiD = theano.shared(WiD_values, name='Wi_D') 
        self.WiD = WiD
        if WfD is None:
            WfD_values = np.zeros((n_in, n_out))
            WfD = theano.shared(WfD_values, name='Wf_D') 
        self.WfD = WfD
        if WcD is None:
            WcD_values = np.zeros((n_in, n_out))
            WcD = theano.shared(WcD_values, name='Wc_D') 
        self.WcD = WcD
        if WoD is None:
            WoD_values = np.zeros((n_in, n_out))
            WoD = theano.shared(WoD_values, name='Wo_D') 
        self.WoD = WoD
        if WtiD is None:
            WtiD_values = np.zeros((n_out, n_out))
            WtiD = theano.shared(WtiD_values, name='Wti_D') 
        self.WtiD = WtiD
        if WtfD is None:
            WtfD_values = np.zeros((n_out, n_out))
            WtfD = theano.shared(WtfD_values, name='Wtf_D') 
        self.WtfD = WtfD
        if WtcD is None:
            WtcD_values = np.zeros((n_out, n_out))
            WtcD = theano.shared(WtcD_values, name='Wtc_D') 
        self.WtcD = WtcD
        if WtoD is None:
            WtoD_values = np.zeros((n_out, n_out))
            WtoD = theano.shared(WtoD_values, name='Wto_D') 
        self.WtoD = WtoD
        if biD is None:
            biD_values = np.zeros(n_out)
            biD = theano.shared(biD_values, name='bi_D') 
        self.biD = biD
        if bfD is None:
            bfD_values = np.zeros(n_out)
            bfD = theano.shared(bfD_values, name='bf_D') 
        self.bfD = bfD
        if bcD is None:
            bcD_values = np.zeros(n_out)
            bcD = theano.shared(bcD_values, name='bc_D') 
        self.bcD = bcD
        if boD is None:
            boD_values = np.zeros(n_out)
            boD = theano.shared(boD_values, name='bo_D') 
        self.boD = boD
        
        ## Initialize deltas for t = 0 cell and hidden states
        if c0D is None:
            c0D_values = np.zeros((n_out,), dtype=theano.config.floatX)
            c0D = theano.shared(value=c0D_values, name="c0D", borrow=True)
        self.c0D = c0D
        if h0D is None:
            h0D_values = np.zeros((n_out,), dtype=theano.config.floatX)
            h0D = theano.shared(value=h0D_values, name="h0D", borrow=True)
        self.h0D = h0D
                  
        self.deltas = [self.WiD, self.WfD, self.WcD, self.WoD, 
                       self.WtiD, self.WtfD, self.WtcD, self.WtoD,
                       self.biD, self.bfD, self.bcD, self.boD, self.c0D, self.h0D]
                      
        ## Initialize peephole deltas
        if peephole_input:
            if WciD is None:
                WciD_values = np.zeros((n_out, n_out))
                WciD = theano.shared(WciD_values, name='Wci_D') 
            self.WciD = WciD
            self.deltas.append(self.WciD)
        if peephole_forget:
            if WcfD is None:
                WcfD_values = np.zeros((n_out, n_out))
                WcfD = theano.shared(WcfD_values, name='Wcf_D') 
            self.WcfD = WcfD
            self.deltas.append(self.WcfD)
        if peephole_output:
            if WcoD is None:
                WcoD_values = np.zeros((n_out, n_out))
                WcoD = theano.shared(WcoD_values, name='Wco_D') 
            self.WcoD = WcoD
            self.deltas.append(self.WcoD)
            
        ## Compute recurrence
        def recurrence(input_t, cell_tm1, hidden_tm1):
            linear_input = T.dot(input_t, self.Wi) + T.dot(hidden_tm1, self.Wti) + self.bi
            if peephole_input:
                linear_input = linear_input + T.dot(cell_tm1, self.Wci)
            input_gate_t = inner_activation(linear_input)
            linear_cell_candidate = T.dot(input_t, self.Wc) + T.dot(hidden_tm1, self.Wtc) + self.bc
            cell_candidate_t = activation(linear_cell_candidate)
            linear_forget = T.dot(input_t, self.Wf) + T.dot(hidden_tm1, self.Wtf) + self.bf
            if peephole_forget:
                linear_forget = linear_forget + T.dot(cell_tm1, self.Wcf)
            forget_gate_t = inner_activation(linear_forget)
            cell_t = input_gate_t * cell_candidate_t + forget_gate_t * cell_tm1
            linear_output = T.dot(input_t, self.Wo) + T.dot(hidden_tm1, self.Wto) + self.bo
            if peephole_output:
                linear_output = linear_output + T.dot(cell_t, self.Wco)
            output_gate_t = inner_activation(linear_output)
            hidden_t = output_gate_t * activation(cell_t)
            return [cell_t, hidden_t]
            
        [cell, hidden], _ = theano.scan(
                                fn=recurrence,
                                sequences=input[::-1] if inverse else input,
                                outputs_info=[T.tile(self.c0,(batch_size, 1)), T.tile(self.h0,(batch_size, 1))],
                                n_steps=input.shape[0]
                                )
        
        if sequence:
            self.output = hidden
        else:
            self.output = hidden[0] if inverse else hidden[-1]
            
        self.input = input
        
        
class GRU:
    '''
    Gated Recurrent Unit layer
    '''
    def __init__(self, rng, input, n_in, n_out, 
                 Wr=None, Wu=None, Wc=None,
                 Wtr=None, Wtu=None, Wtc=None,
                 br=None, bu=None, bc=None, h0=None,
                 WrD=None, WuD=None, WcD=None,
                 WtrD=None, WtuD=None, WtcD=None,
                 brD=None, buD=None, bcD=None, h0D=None,
                 batch_size=1, activation_hidden=T.nnet.sigmoid,
                 sequence=False, inverse=False): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int, int, theano.Op
        """ 
        Initialize parameters 
        """        
        ## Initialize weights uniformely sampled from (low, high) interval as a n_in X n_out matrix
        if Wu is None:
            Wu_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wu = theano.shared(value=Wu_values, name="Wu", borrow=True)
        self.Wu = Wu

        if Wtu is None:
            Wtu_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wtu = theano.shared(value=Wtu_values, name="Wtu", borrow=True)
        self.Wtu = Wtu

        if bu is None:
            bu_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bu = theano.shared(value=bu_values, name="bu", borrow=True)
        self.bu = bu

        if Wr is None:
            Wr_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wr = theano.shared(value=Wr_values, name="Wr", borrow=True)
        self.Wr = Wr

        if Wtr is None:
            Wtr_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wtr = theano.shared(value=Wtr_values, name="Wtr", borrow=True)
        self.Wtr = Wtr

        if br is None:
            br_values = np.zeros((n_out,), dtype=theano.config.floatX)
            br = theano.shared(value=br_values, name="br", borrow=True)
        self.br = br
                
        if Wc is None:
            Wc_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_in, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wc = theano.shared(value=Wc_values, name="Wc", borrow=True)
        self.Wc = Wc
                
        if Wtc is None:
            Wtc_values = np.asarray(
                                  rng.normal(scale = .01, size = (n_out, n_out)),
                                  dtype = theano.config.floatX
                                  )
            Wtc = theano.shared(value=Wtc_values, name="Wtc", borrow=True)
        self.Wtc = Wtc

        if bc is None:
            bc_values = np.zeros((n_out,), dtype=theano.config.floatX)
            bc = theano.shared(value=bc_values, name="bc", borrow=True)
        self.bc = bc

        ## Initialize hidden state for t = 0
        if h0 is None:
            h0_values = np.zeros((n_out,), dtype=theano.config.floatX)
            h0 = theano.shared(value=h0_values, name="h0", borrow=True)
        self.h0 = h0
        
        self.params = [self.Wr, self.Wu, self.Wc, 
                       self.Wtr, self.Wtu, self.Wtc,
                       self.br, self.bu, self.bc, self.h0]

        ## Initialize deltas
        if WrD is None:
            WrD_values = np.zeros((n_in, n_out))
            WrD = theano.shared(WrD_values, name='Wr_D') 
        self.WrD = WrD
        if WuD is None:
            WuD_values = np.zeros((n_in, n_out))
            WuD = theano.shared(WuD_values, name='Wu_D') 
        self.WuD = WuD
        if WcD is None:
            WcD_values = np.zeros((n_in, n_out))
            WcD = theano.shared(WcD_values, name='Wc_D') 
        self.WcD = WcD
        if WtrD is None:
            WtrD_values = np.zeros((n_out, n_out))
            WtrD = theano.shared(WtrD_values, name='Wtr_D') 
        self.WtrD = WtrD
        if WtuD is None:
            WtuD_values = np.zeros((n_out, n_out))
            WtuD = theano.shared(WtuD_values, name='Wtu_D') 
        self.WtuD = WtuD
        if WtcD is None:
            WtcD_values = np.zeros((n_out, n_out))
            WtcD = theano.shared(WtcD_values, name='Wtc_D') 
        self.WtcD = WtcD
        if brD is None:
            brD_values = np.zeros(n_out)
            brD = theano.shared(brD_values, name='br_D') 
        self.brD = brD
        if buD is None:
            buD_values = np.zeros(n_out)
            buD = theano.shared(buD_values, name='bu_D') 
        self.buD = buD
        if bcD is None:
            bcD_values = np.zeros(n_out)
            bcD = theano.shared(bcD_values, name='bc_D') 
        self.bcD = bcD
        if h0D is None:
            h0D_values = np.zeros((n_out,), dtype=theano.config.floatX)
            h0D = theano.shared(value=h0D_values, name="h0D", borrow=True)
        self.h0D = h0D
                  
        self.deltas = [self.WrD, self.WuD, self.WcD,
                       self.WtrD, self.WtuD, self.WtcD,
                       self.brD, self.buD, self.bcD, self.h0D]

        ## Compute recurrence
        def recurrence(input_t, hidden_tm1):
            linear_reset = T.dot(input_t, self.Wr) + T.dot(hidden_tm1, self.Wtr) + self.br
            reset_t = T.nnet.sigmoid(linear_reset)
            linear_update = T.dot(input_t, self.Wu) + T.dot(hidden_tm1, self.Wtu) + self.bu
            update_t = T.nnet.sigmoid(linear_update)
            linear_cell = T.dot(input_t, self.Wc) + T.dot(hidden_tm1 * reset_t, self.Wtc) + self.bc
            cell_t = T.tanh(linear_cell)
            hidden_t = (1 - update_t) * cell_t + update_t * hidden_tm1
            return hidden_t
            
        hidden, _ = theano.scan(
                                fn=recurrence,
                                sequences=input[::-1] if inverse else input,
                                outputs_info=T.tile(self.h0,(batch_size, 1)),
                                n_steps=input.shape[0]
                                )
        
        if sequence:
            self.output = hidden
        else:
            self.output = hidden[0] if inverse else hidden[-1]
            
        self.input = input
        

class Attention:
    '''
    Attention layer
    '''
    
    def __init__(self, query, info): # HiddenLayer, numpy.random.RandomState, theano.tensor.dmatrix, int,
    
                 
        ## Compute recurrence
        def recurrence(query_t, info_t):
            aggregation_t = T.dot(info_t, query_t) / T.sqrt(T.sum(T.numpy.multiply(info_t, info_t),axis=0) * T.dot(query_t, query_t))
            relevance_t = T.nnet.softmax(aggregation_t)
            output_t = T.sum(relevance_t * np.transpose(info_t), axis=1)
            return output_t 
                
        output, _ = theano.scan(
                                fn=recurrence,
                                sequences=[query, info],
                                #n_steps=T.shape(query)[0]
                                )
        
        self.output = output
        
        self.query = query
        self.info = info
        