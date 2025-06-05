"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2024 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os, sys, time, warnings, itertools, inspect

from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.metrics import Mean

# legacy
#from .losses import *
#from .logging import *


def plot_piecewise_constant_decay(boundaries, values, max_iteration=600_000):
    # plots the learning rate returned by PiecewiseConstantDecay
    plt.figure(figsize=(12,4))
    plt.plot([0,*np.repeat(boundaries,2),max_iteration], np.repeat(values,2))
    plt.ticklabel_format(style='plain'); plt.ylim(0,None); plt.xlim(0,max_iteration); plt.grid()
    plt.show()

def plot_scheduler(scheduler, max_iteration=600_000):
    # plots the learning rate returned by keras schedulers
    # slow, but works for all schedulers
    x = range(0, max_iteration, 100)
    y = [float(scheduler(i)) for i in x]
    plt.figure(figsize=(12,4))
    plt.plot(x,y)
    plt.ticklabel_format(style='plain'); plt.ylim(0,None); plt.xlim(0,max_iteration); plt.grid()
    plt.show()


class LearningRateDecay(Callback):
    def __init__(self, methode='linear', base_lr=1e-3, n_desired=40000, desired=0.1, bias=0.0, minimum=0.1):
        super(LearningRateDecay, self).__init__()
        self.methode = methode
        self.base_lr = base_lr
        self.n_desired = n_desired
        self.desired = desired
        self.bias = bias
        self.minimum = minimum

        #TODO: better naming

    def compute_learning_rate(self, n, methode):
        n_desired = self.n_desired
        desired = self.desired
        base_lr = self.base_lr
        bias = self.bias

        offset = base_lr * desired * bias
        base_lr = base_lr - offset

        desired = desired / (1-desired*bias) * (1-bias)

        if methode == 'default':
            k = (1 - desired) / n_desired
            lr = np.maximum( -k * n + 1, 0)
        elif methode == 'linear':
            k = (1 / desired - 1) / n_desired
            lr = 1 / (1 + k * n)
        elif methode == 'quadratic':
            k = (np.sqrt(1/desired)-1) / n_desired
            lr = 1 / (1 + k * n)**2
        elif methode == 'exponential':
            k = -1 * np.log(desired) / n_desired
            lr = np.exp(-k*n)

        lr = base_lr * lr + offset
        lr = np.maximum(lr, self.base_lr * self.minimum)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        steps_per_epoch = self.params['steps']
        iteration = self.epoch * steps_per_epoch + batch

        lr = self.compute_learning_rate(iteration, self.methode)
        K.set_value(self.model.optimizer.lr, lr)

    def plot_learning_rates(self):
        n = np.linspace(0, self.n_desired*2, 101)
        plt.figure(figsize=[16, 6])
        plt.plot([n[0], n[-1]], [self.base_lr*self.desired*self.bias]*2, 'k')
        for m in ['default', 'linear', 'quadratic', 'exponential']:
            plt.plot(n, self.compute_learning_rate(n, m))
        plt.legend(['bias', '$-kn+1$', '$1/(1+kn)$', '$1/(1+kn)^2$', '$e^{-kn}$'])
        plt.grid()
        plt.xlim(0, n[-1])
        plt.ylim(0, None)
        plt.show()


class ModelSnapshot(Callback):
    """Save the model weights after an interval of iterations."""

    def __init__(self, logdir, interval=10000, verbose=1):
        super(ModelSnapshot, self).__init__()
        self.logdir = logdir
        self.interval = interval
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # steps/batches/iterations
        steps_per_epoch = self.params['steps']
        self.iteration = self.epoch * steps_per_epoch + batch + 1

    def on_batch_end(self, batch, logs=None):
        if self.iteration % self.interval == 0:
            filepath = os.path.join(self.logdir, 'weights.%06i.h5' % (self.iteration))
            if self.verbose > 0:
                print('\nSaving model %s' % (filepath))
            self.model.save_weights(filepath, overwrite=True)


class Logger(Callback):

    def __init__(self, logdir):
        super(Logger, self).__init__()
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    def save_history(self):
        df = pd.DataFrame.from_dict(self.model.history.history)
        df.to_csv(os.path.join(self.logdir, 'history.csv'), index=False)

    def append_log(self, logs):
        data = {k:[float(logs[k])] for k in self.model.metrics_names}
        data['iteration'] = [self.iteration]
        data['epoch'] = [self.epoch]
        data['batch'] = [self.batch]
        data['time'] = [time.time() - self.start_time]
        #data['lr'] = [float(K.get_value(self.model.optimizer.lr))]
        df = pd.DataFrame.from_dict(data)
        with open(os.path.join(self.logdir, 'log.csv'), 'a') as f:
            df.to_csv(f, header=f.tell()==0, index=False)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.save_history()

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # steps/batches/iterations
        steps_per_epoch = self.params['steps']
        self.iteration = self.epoch * steps_per_epoch + batch

    def on_batch_end(self, batch, logs=None):
        self.append_log(logs)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.save_history()


# obsolete
class AdamAccumulate(Optimizer):
    """Adam optimizer with accumulated gradients for having a virtual batch size larger
    than the physical batch size.

    # Notes
        Default parameters follow those provided in the original paper.
        Only works with TensorFlow 1.x!
        Keras > 1.x comes with gradient accumulation

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        accum_iters: Number of batches between parameter update.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
    """
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, accum_iters=10, **kwargs):
        super(AdamAccumulate, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.accum_iters = K.variable(accum_iters)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = ms + vs

        flag = K.equal(t % self.accum_iters, 0)
        flag = K.cast(flag, dtype='float32')

        for p, g, m, v, gg in zip(params, grads, ms, vs, gs):

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + flag * g) / self.accum_iters)
            p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, flag * m_t + (1 - flag) * m))
            self.updates.append((v, flag * v_t + (1 - flag) * v))
            self.updates.append((gg, gg_t))

            # apply constraints.
            new_p = p_t
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

