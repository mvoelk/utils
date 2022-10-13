"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import matplotlib.pyplot as plt


def bar_plot(labels, data, title=None, total_width=.8, single_width=.9, vertical=False):
    '''
    # Arguments
        labels: list of strings etc.
        data: dict of data, key is used for legend
        title: string
    '''

    plt.figure(figsize=(16,16))
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

    ax.legend(bars, data.keys())

    if vertical:
        ax.set_xticks(list(range(len(product_names))))
        ax.set_xticklabels(product_names, rotation=90)
    else:
        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(labels, rotation=0)

    plt.title(title)
    plt.show()
