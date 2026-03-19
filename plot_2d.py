"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import matplotlib.pyplot as plt


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
        plot_abs_error_hist(y_true[i,...,1], y_pred[i,...,1], bins=100, y_max=100, cumulative=False)

    # Example 2
        plot_abs_error_hist(y_true[i,...,1], y_pred[i,...,1], mask, bins=100, y_max=None, cumulative=True)
    """
    
    import seaborn as sns

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
    '''Error plot (mean and std) for a batch of features

    # Arguments
        y: shape (..., num_features)
        mask: shape (...)
    '''
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


def plot_points(points1, points2=None, T1=None, T2=None,
                view=(225.0, 22.5), point_size=2.0, figsize=(12,6), limits=(-1,1), with_normals=True):
    '''Plots one or two point clouds side by side

    # Arguments
        points1, points2: shape (num_points, 3) for xyz values, or (num_points, 6) with normals
        T1, T2: homogenous transformations for frame visualization
        view: tuple with azimuth and elevation in degree, or preset string 'x', 'y', 'z'
    '''

    proj_type = 'ortho' if view in ['x', 'y', 'z'] else 'persp'

    view_presets = {
        'x': ( 0.0,  0.0),
        'y': (90.0,  0.0),
        'z': (90.0, 90.0),
    }
    if isinstance(view, str) and view in view_presets:
        view = view_presets[view]
    
    def new_subplot(pos, points, T):
        ax = plt.subplot(pos, projection='3d', proj_type=proj_type)
        ax.scatter(*points[...,:3].T, s=point_size)
        
        if with_normals and points.shape[-1] == 6:
            ax.quiver(*points[...,:3].T, *points[...,3:6].T, alpha=0.3, length=0.02*point_size, color='tab:red')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_zlim(*limits)
        ax.set_box_aspect(aspect = (1,1,1))
        ax.view_init(azim=view[0], elev=view[1])

        if T is not None:
            draw_frame_3d(T, length=0.5)
    
    fig = plt.figure(figsize=figsize)

    if points2 is None:
        new_subplot(111, points1, T1)
    else:
        new_subplot(121, points1, T1)
        new_subplot(122, points2, T2)

    try:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
    except:
        pass

    plt.tight_layout()
    plt.show()


def plot_point_assignment(points1, points2, ordered=False, point_size=2.0, figsize=(10,10), limits=(-1,1)):
    
    n1, c = points1.shape
    n2, c = points2.shape

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    if not ordered:
        # use Chamfer Distance
        ds = np.sum((points1[:,None,:] - points2[None,:,:]) ** 2, axis=-1) ** 0.5
        #dist1, dist2 = np.min(ds, axis=-1), np.min(ds, axis=-2)
        idx1, idx2 = np.argmin(ds, axis=-1), np.argmin(ds, axis=-2)
        
        mask1, mask2 = np.zeros(n2, dtype='bool'), np.zeros(n1, dtype='bool')
        mask1[idx1], mask2[idx2] = True, True
        
        ax.scatter(*points1[mask2].T, s=point_size, color='tab:blue', alpha=0.6)
        ax.scatter(*points1[~mask2].T, s=point_size, color='tab:green', alpha=0.8)
        ax.scatter(*points2[mask1].T, s=point_size, color='tab:red', alpha=0.6)
        ax.scatter(*points2[~mask1].T, s=point_size, color='tab:orange', alpha=0.8)
        ax.quiver(*points1.T, *(points2[idx1]-points1).T, alpha=0.3, color='tab:blue')
        ax.quiver(*points2.T, *(points1[idx2]-points2).T, alpha=0.3, color='tab:red')

    else:
        ax.scatter(*points1.T, s=point_size, color='tab:blue', alpha=0.6)
        ax.scatter(*points2.T, s=point_size, color='tab:red', alpha=0.6)
        ax.quiver(*points1.T, *(points2-points1).T, alpha=0.3, color='tab:gray', arrow_length_ratio=0)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_zlim(*limits)

    try:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
    except:
        pass

    plt.tight_layout()
    plt.show()


def draw_bbox(box, linewidth=1, edgecolor='r'):
    '''Draws a axis aligned bounding box'''
    x, y, w, h = box
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((x-1, y-1), w+1, h+1, linewidth=linewidth, edgecolor=edgecolor, facecolor='none'))

def draw_frame_3d(T=np.eye(4), length=1.0, lw=2):
    '''Draws a homogeneous transformation matrix in 3d'''
    R, t = T[:3,:3], T[:3,3]
    axes = R * length
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    ax = plt.gca()
    for i in range(3):
        ax.quiver(*t, *axes[:,i], color=colors[i], linewidth=lw)
        ax.text(*(t + axes[:,i]), labels[i], color=colors[i])


def plot_tiled_images(imgs, gap=1, figsize=(9,8), colorbar=False, fname=None, cmap='viridis', vmin=None, vmax=None, title=None):

    # imgs: shape (n_rows, n_cols, h, w)
    # fname: string

    n_rows, n_cols, h, w = imgs.shape

    H = n_rows * h + (n_rows - 1) * gap
    W = n_cols * w + (n_cols - 1) * gap
    
    canvas = np.full((H, W), np.nan, dtype='float32')
    for r in range(n_rows):
        for c in range(n_cols):
            i0 = r * (h + gap)
            j0 = c * (w + gap)
            canvas[i0:i0 + h, j0:j0 + w] = imgs[r, c]
    
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='white')
    
    vmin = np.min(imgs) if vmin is None else vmin
    vmax = np.max(imgs) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(canvas), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', origin='upper')
    ax.axis('off'); ax.set_title(title)

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

