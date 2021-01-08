#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:41:23 2020

@author: mdeb
"""
import numpy as np

class Adam():
    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., amsgrad=True):
        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.amsgrad = amsgrad
        self.initial_decay = decay
#        if amsgrad:
#            self.epsilon = 0.
#        else:
        self.epsilon = epsilon
    def get_update(self, grad):
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))
        if not hasattr(self, 'ms'):
            self.ms = np.zeros(grad.shape)
            self.vs = np.zeros(grad.shape)
        m_t = (self.beta_1 * self.ms) + (1. - self.beta_1) * grad
        v_t = (self.beta_2 * self.vs) + (1. - self.beta_2) * np.square(grad)
        if self.amsgrad:
            v_t = np.maximum(v_t, self.vs)
#        update = lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
        update = lr_t * m_t / np.sqrt(v_t)
        self.ms = m_t
        self.vs = v_t
        self.iterations += 1
        return update