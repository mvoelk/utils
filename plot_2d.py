"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bar_plot(data, labels=None, title=None, total_width=.8, single_width=.9, vertical=False, figsize=(16,16), lim=(None,None), loc='best'):
    '''
    # Arguments
        data: dict of data series, key is used for legend
        labels: list of strings etc., tick labels
        title: string
    '''

    plt.figure(figsize=figsize)
    ax = plt.gca()

    n_bars = len(data)
    bar_width = total_width / n_bars

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bars = []

    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            if vertical:
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
            else:
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[i % len(colors)])

        bars.append(bar[0])

    ax.legend(bars, data.keys(), loc=loc)

    if labels is not None:
        if vertical:
            ax.set_xticks(list(range(len(labels))))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_xlim(-1, len(labels))
            ax.set_ylim(*lim)
        else:
            ax.set_yticks(list(range(len(labels))))
            ax.set_yticklabels(labels, rotation=0)
            ax.set_ylim(-1, len(labels))
            ax.set_xlim(*lim)

    plt.title(title)
    plt.show()


def plot_abs_error_hist(y_true, y_pred=None, mask=None, bins=100, cumulative=False, name='', x_max=None, y_max=None):
    """Plots the histogram of the absolute prediction error.

    # Notes
        y_true and y_pred are flattened
        mask has to be of same shape

    # Example 1
        mask = y_true[i,...,0] < 0.5
        plot_abs_error_hist(y_true[i,...,1], y_pred[i,...,1], bins=100, y_max=100, cumulative=False)
        plot_abs_error_hist(y_true[i,...,1], y_pred[i,...,1], mask, bins=100, y_max=100, cumulative=False)
    # Example 2
        mask = y_true[i,...,0] < 0.5
        plot_abs_error_hist(y_true[i,...,1], y_pred[i,...,1], bins=100, y_max=None, cumulative=True)
        plot_abs_error_hist(y_true[i,...,1], y_pred[i,...,1], mask, bins=100, y_max=None, cumulative=True)
    """
    
    y_true = np.reshape(np.float32(y_true), (-1))
    
    if y_pred is not None:
        y_pred = np.reshape(np.float32(y_pred), (-1))
    else:
        y_pred = 0
    
    abs_err = np.abs(y_true-y_pred)

    if mask is not None:
        mask = np.reshape(np.bool8(mask), (-1))
        abs_err = abs_err[mask]

    plt.figure(figsize=(16,4))
    sns.histplot(abs_err, kde=True, bins=bins, cumulative=cumulative)
    plt.ylim(0, y_max); plt.xlim(0, x_max)
    plt.axvline(x=np.mean(abs_err), color='red')
    sparsity = 1-len(abs_err)/len(y_true)
    plt.title(name + (' (sparsity %.2f)'%(sparsity) if sparsity > 0 else ''))
    plt.show()


def plot_output_distribution(y, mask=None):
    # y: shape (..., num_features)
    # mask: shape (...)
    
    num_features = y.shape[-1]
    a = np.reshape(y, (-1,num_features))
    if mask is not None:
        m = np.reshape(mask, (-1))
        a = a[m]

    plt.figure(figsize=(16,4))
    r = range(num_features)
    plt.errorbar(r, np.mean(a, axis=0), np.std(a, axis=0), linestyle='None', marker='x')
    plt.xticks(r)
    plt.xlim(-1, num_features)
    plt.grid()
    plt.show()
