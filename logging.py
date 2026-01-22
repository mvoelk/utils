"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2024 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, time, warnings, itertools


class MetricUtility():
    """History and Log for Tensorflow 2.x Eager API.
    
    # Arguments
        names: list of metric names.
        logdir: If specified, the log and history are written to csv files.
    
    The update method receives a dictionary with values for each metric and
    should be called after each iteration.
    
    # Example
        mu = MetricUtility(['loss', 'accuracy'], logdir='./')
        for each epoch:
            mu.on_epoch_begin()
            for each training step:
                ...
                mu.update(metric_values, training=True)
            for each validation step:
                ...
                mu.update(metric_values, training=False)
            mu.on_epoch_end(verbose=True)
    """
    
    def __init__(self, names=['loss',], logdir=None, optimizer=None):
        self.names = names
        self.logdir = logdir
        self.optimizer = optimizer
        
        if logdir is not None:
            self.log_path = os.path.join(self.logdir, 'log.csv')
            self.history_path = os.path.join(self.logdir, 'history.csv')
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
            if os.path.exists(self.history_path):
                os.remove(self.history_path)
        
        self.reset()
        
    def reset(self):
        self.iteration = 0
        self.epoch = 0
        self.log = {n: [] for n in self.names}
        self.log.update({'epoch': [], 'time': []})
        self.history = {n: [] for n in self.names}
        self.history.update({'val_'+n: [] for n in self.names})
        self.history.update({'epoch': [], 'time': []})
        if self.optimizer is not None:
            self.log.update({'learning_rate': []})
            self.history.update({'learning_rate': []})
    
    def on_epoch_begin(self):
        from keras.metrics import Mean
        
        if self.epoch == 0:
            self.t0 = time.time()
        self.t1 = time.time()
        self.epoch += 1
        self.steps = 0
        self.steps_val = 0
        self.metrics = {n: Mean() for n in self.names}
        self.metrics_val = {n: Mean() for n in self.names}
        
    def update(self, values, training=True):
        float_values = {n: float(v) for n, v in values.items()}
        if training:
            self.t2 = time.time()
            self.iteration += 1
            self.steps += 1
            for n, v in float_values.items():
                self.metrics[n].update_state(v)
                self.log[n].append(v)
            self.log['epoch'].append(self.epoch)
            self.log['time'].append(time.time()-self.t0)
            if self.optimizer is not None:
                self.log['learning_rate'].append(float(self.optimizer.learning_rate))
            
            if self.logdir is not None:
                float_values = {k:v[-1:] for k, v in self.log.items()}
                df = pd.DataFrame.from_dict(float_values)
                with open(self.log_path, 'a') as f:
                    df.to_csv(f, header=f.tell()==0, index=False)
        else:
            self.steps_val += 1
            for n, v in float_values.items():
                self.metrics_val[n].update_state(v)

    def on_epoch_end(self, verbose=True):
        if self.steps == 0:
            warnings.warn('no metric update was done')
            return
        
        float_values = {n: float(m.result()) for n, m in self.metrics.items()}
        if self.steps_val > 0:
            float_values.update({'val_'+n: float(m.result()) for n, m in self.metrics_val.items()})
        
        for n, v in float_values.items():
            self.history[n].append(v)
        self.history['epoch'].append(self.epoch)
        self.history['time'].append(time.time()-self.t0)
        if self.optimizer is not None:
            self.history['learning_rate'].append(float(self.optimizer.learning_rate))
        
        if self.logdir is not None:
            float_values = {k:v[-1:] for k, v in self.history.items() if len(v)}
            df = pd.DataFrame.from_dict(float_values)
            with open(self.history_path, 'a') as f:
                df.to_csv(f, header=f.tell()==0, index=False)
        
        if verbose:
            t1, t2, t3 = self.t1, self.t2, time.time()
            for n, v in self.history.items():
                if len(v):
                    print('%s %5.5f ' % (n, v[-1]), end='')
            print('\n%.1f minutes/epoch  %.2f iter/sec' % ((t3-t1)/60, self.steps/(t2-t1)))


def filter_signal(x, y, window_length=1000):
    if type(window_length) is not int or len(y) <= window_length:
        return [], []
    
    #w = np.ones(window_length) # moving average
    w = np.hanning(window_length) # hanning window
    
    wlh = int(window_length/2)
    if x is None:
        x = np.arange(wlh, len(y)-wlh+1)
    else:
        x = x[wlh:len(y)-wlh+1]
    y = np.convolve(w/w.sum(), y, mode='valid')
    return x, y


def plot_log(log_dirs, names=None, limits=None, window_length=250,
             filtered_only=False, autoscale=True, save_plots=False):
    """Plot and compares the training log contained in './checkpoints/'.
    
    # Arguments
        log_dirs: string or list of string with directory names.
        names: list of strings with metric names in 'log.csv'.
            None means all.
        limits: tuple with min and max iteration that should be plotted.
            None means no limits.
        window_length: int, window length for signal filter.
            None means no filtered signal is plotted.
    
    # Notes
        The epoch is inferred from the log with the most iterations.
        Different batch size leads to different epoch length.
    """
    
    loss_terms = {'loss', 'error', 'err', 'abs', 'sqr', 'dist', 'reg'}
    metric_terms = {'precision', 'recall', 'fmeasure', 'accuracy', 'sparsity', 'visibility'}
    
    if save_plots:
        plotdir = './plots/' + time.strftime('%Y%m%d%H%M') + '_log'
        os.makedirs(plotdir, exist_ok=True)
    
    if type(log_dirs) == str:
        log_dirs = [log_dirs]
    log_dirs = list(log_dirs)
    for d in [d for d in log_dirs]:
        if not os.path.isfile(os.path.join('.', 'checkpoints', d, 'log.csv')):
            print(d+' not found')
            log_dirs.remove(d)
    
    if limits is None:
        limits = slice(None)
    elif type(limits) in [list, tuple]:
        limits = slice(*limits)
    
    dfs = []
    max_df = []
    all_names = set()
    for d in log_dirs:
        csv_path = os.path.join('.', 'checkpoints', d, 'log.csv')
        #df = pd.read_csv(csv_path)
        try:
            df = pd.read_csv(csv_path, engine='pyarrow', dtype='float32')
        except:
            df = pd.read_csv(csv_path)
        all_names.update(df.keys())
        if 'iteration' not in df.keys():
            df['iteration'] = np.arange(1,len(df)+1)
        if 'epoch' not in df.keys():
            df['epoch'] = np.zeros(len(df))
        if len(df) >= len(max_df):
            max_df = df
        df = df[limits]
        df = {k: np.array(df[k]) for k in df.keys()}
        dfs.append(df)

    if names is None:
        names = all_names.difference({'time', 'epoch', 'iteration'})
        print(names)

    iteration = np.int32(max_df['iteration'])
    epoch = np.int32(max_df['epoch'])
    idx = np.argwhere(np.diff(epoch))[:,0]

    min_lim, max_lim = limits.start or 0, limits.stop or len(iteration)

    idx_red = idx
    if len(idx) > 1:
        steps_per_epoch = int(idx[1]-idx[0])
        print('steps per epoch %i ' % (steps_per_epoch))
        if 'time' in max_df.keys():
            seconds_per_epoch = (max_df['time'][idx[1]]-max_df['time'][idx[0]]) / 3600
            print('time per epoch %3.1f h' % (seconds_per_epoch))

        # reduce epoch ticks
        max_ticks = 10
        step = 1
        while True:
            n = np.sum((idx_red > min_lim) & (idx_red < max_lim))
            if n < max_ticks:
                break
            elif n < 2*max_ticks:
                step *= 2
            elif n < 5*max_ticks:
                step *= 5
            elif n < 10*max_ticks:
                step *= 10
            idx_red = idx_red[step-1::step]

    colorgen = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = [next(colorgen) for i in range(len(dfs))]
    
    for k in names:
        plt.figure(figsize=(16, 6))
        xmin, xmax, ymin, ymax, ymean = 2147483647, -2147483648, np.inf, -np.inf, 0.0
        yref1, yref2 = 1e-8, 0.0
        found = False
        for i, df in enumerate(dfs):
            if k in df.keys() and len(df[k]):
                x, y = df['iteration'], df[k]
                
                yref1 += np.mean(y[:100])
                yref2 += np.mean(y[-100:])
                found = True

                finite = np.all(np.isfinite(y))
                label = log_dirs[i] + ' (NaN)'*(not finite)

                if window_length:
                    x_, y_ = filter_signal(x, y, window_length)
                    plt.plot(x_, y_, color=colors[i], label=label)
                    if not filtered_only:
                        plt.plot(x, y, zorder=0, color=colors[i], alpha=0.3)
                    else:
                        y = y_
                else:
                    plt.plot(x, y, zorder=0, color=colors[i], label=label)

                xmin, xmax = min(xmin, x[0]), max(xmax, x[-1])
                if len(y) and finite:
                    ymin, ymax, ymean = min(ymin, np.min(y)), max(ymax, np.max(y)), max(ymean, np.mean(y))
        
        if found:
            #legend_loc = 'best' # nice but slow as hell
            legend_loc = 'upper right' if yref1 > yref2 else 'lower right'

            plt.title(k, y=1.05)
            plt.legend(loc=legend_loc)
            
            ax1 = plt.gca()
            ax1.set_xlim(xmin, xmax)
            if autoscale:
                k_split = k.split('_')
                if len(loss_terms.intersection(k_split)):
                    plt.ylim(0, min(ymax*1.05, ymean*4))
                elif len(metric_terms.intersection(k_split)):
                    plt.ylim(0, 1)
            ax1.yaxis.grid(True)
            #ax1.set_xlabel('iteration')
            #ax1.set_yscale('linear')
            ax1.get_yaxis().get_major_formatter().set_useOffset(False)
            
            ax2 = ax1.twiny()
            ax2.xaxis.grid(True)
            ax2.set_xticks(iteration[idx_red])
            ax2.set_xticklabels(epoch[idx_red])
            ax2.set_xlim(xmin, xmax)
            #ax2.set_xlabel('epoch')
            #ax2.set_yscale('linear')
            ax2.get_yaxis().get_major_formatter().set_useOffset(False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(plotdir+'/%s.pdf'%(k), bbox_inches='tight')
            plt.show()
            
        else:
            #print(k+' no values')
            plt.close()

def plot_history(log_dirs, names=None, limits=None, autoscale=True, validation=True, save_plots=False, markers=False):

    loss_terms = {'loss', 'error', 'err', 'abs', 'sqr', 'dist', 'reg'}
    metric_terms = {'precision', 'recall', 'fmeasure', 'accuracy', 'sparsity', 'visibility'}
    
    if save_plots:
        plotdir = './plots/' + time.strftime('%Y%m%d%H%M') + '_history'
        os.makedirs(plotdir, exist_ok=True)
    
    if type(log_dirs) == str:
        log_dirs = [log_dirs]
    log_dirs = list(log_dirs)
    for d in [d for d in log_dirs]:
        if not os.path.isfile(os.path.join('.', 'checkpoints', d, 'history.csv')):
            print(d+' not found')
            log_dirs.remove(d)
    
    if limits is None:
        limits = slice(None)
    elif type(limits) in [list, tuple]:
        limits = slice(*limits)
    
    dfs = []
    max_df = []
    all_names = set()
    for d in log_dirs:
        df = pd.read_csv(os.path.join('.', 'checkpoints', d, 'history.csv'))
        all_names.update(df.keys())
        if len(df) > len(max_df):
            max_df = df
        if 'epoch' not in df.keys():
            df['epoch'] = np.arange(1,len(df)+1)
        df = df[limits]
        df = {k: np.array(df[k]) for k in df.keys()}
        dfs.append(df)
    
    if names is None:
        names = {n for n in all_names if not n.startswith('val_')}
        names = names.difference({'time', 'epoch'})
        print(names)
    
    colorgen = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = [next(colorgen) for i in range(len(dfs))]
    
    for k in names:
        plt.figure(figsize=(16,4))
        
        xmin, xmax = 2147483647, 0
        ymin, ymax = sys.float_info.max, sys.float_info.min
        for i, df in enumerate(dfs):
            if len(df['epoch']):
                xmin, xmax = min(xmin, df['epoch'][0]), max(xmax, df['epoch'][-1])
            if k in df.keys() and len(df[k]):
                if np.all(np.isfinite(df[k])):
                    if autoscale:
                        ymin, ymax = min(ymin, np.min(df[k])), max(ymax, np.max(df[k]))
                    label = log_dirs[i]
                else:
                    label = log_dirs[i] + ' (NaN)'
                plt.plot(df['epoch'], df[k], '-o' if markers else '-', color=colors[i], label=label, markersize=5)

            kv = 'val_'+k
            if validation and kv in df.keys() and len(df[kv]):
                if np.all(np.isfinite(df[kv])):
                    if autoscale:
                        ymin, ymax = min(ymin, np.max(df[kv])), max(ymax, np.max(df[kv]))
                plt.plot(df['epoch'], df[kv], '--o' if markers else '--', color=colors[i], markersize=5)
        
        if ymax > sys.float_info.min:
            epoch = list(range(0,xmax+1))
            ax = plt.gca()
            ax.set_xticks(epoch)
            ax.set_xticklabels(epoch)
            plt.xlim(xmin, xmax)
            if autoscale:
                k_split = k.split('_')
                if len(loss_terms.intersection(k_split)):
                    plt.ylim(0, ymax*1.05)
                elif len(metric_terms.intersection(k_split)):
                    #plt.ylim(0, 1)
                    plt.ylim(np.floor(ymin*10)/10, np.ceil(ymax*10)/10)
                    #plt.hlines([0.5,0.8,0.9], xmin, xmax, linestyles='-.', linewidth=1)
            plt.title(k)
            plt.grid()
            plt.legend(loc='best')
            plt.tight_layout()
            if save_plots:
                plt.savefig(plotdir+'/%s.pdf'%(k), bbox_inches='tight')
            plt.show()
        else:
            #print(k+' no values')
            plt.close()

