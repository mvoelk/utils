"""Some utils related to Keras models.

SPDX-License-Identifier: MIT
Copyright © 2017 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/ssd_detectors
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from keras import backend as K
from keras.models import Model


def get_layers(model):
    """Collects all layers in a model and its nested models."""
    layers = []
    def get(m):
        for l in m.layers:
            if l.__class__.__name__ in ['Model', 'Functional']:
                get(l)
            else:
                if l not in layers:
                    layers.append(l)
    get(model)
    return layers


def load_weights(model, filepath, layer_names=None):
    """Loads layer weights from a HDF5 save file by name.

    # Arguments
        model: Keras model
        filepath: Path to HDF5 file
        layer_names: List of strings, names of the layers for which the
            weights should be loaded. List of tuples
            (name_in_file, name_in_model), if the names in the file differ
            from those in model.
    """
    filepath = os.path.expanduser(filepath)
    f = h5py.File(filepath, 'r')

    if layer_names == None:
        layer_names = f.attrs['layer_names']

    for name in layer_names:
        if type(name) in [tuple, list]:
            name_model = name[1]
            name_file = name[0]
        else:
            name_model = str(name, 'utf-8')
            name_file = name
        g = f[name_file]
        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
        try:
            layer = model.get_layer(name_model)
            #assert layer is not None
        except ValueError:
            print('layer missing %s' % (name_model))
            print('    file  %s' % ([w.shape for w in weights]))
            continue
        try:
            #print('load %s' % (name_model))
            layer.set_weights(weights)
        except Exception as e:
            print('something went wrong %s' % (name_model))
            print('    model %s' % ([w.shape.as_list() for w in layer.weights]))
            print('    file  %s' % ([w.shape for w in weights]))
            print(e)
    f.close()


def freeze_layers(model, trainable_conv_layers=0, trainable_bn_layers=0):
    """Set layers to none trainable.

    # Argumentes
        model: Keras model
        trainable_conv_layers: Number ob trainable convolution layers at 
            the end of the architecture.
        trainable_bn_layers: Number ob trainable batchnorm layers at the 
            end of the architecture.
    """
    layers = [l for l in model.layers if l.__class__.__name__ in ['Dense', 'Conv1D', 'Conv2D', 'Conv3D']]
    for i, l in enumerate(layers[::-1]):
        l.trainable = i < trainable_conv_layers

    layers = [l for l in model.layers if l.__class__.__name__ in ['BatchNormalization']]
    for i, l in enumerate(layers[::-1]):
        l.trainable = i < trainable_bn_layers


def calc_memory_usage(model, batch_size=1):
    """Compute the memory usage of a keras modell.

    # Arguments
        model: Keras model.
        batch_size: Batch size used for training.

    source: https://stackoverflow.com/a/46216013/445710
    """
    # TODO: nested models, recursion

    shapes_mem_count = 0
    #shapes_mem_count += np.sum([np.sum([np.sum([np.prod(s[1:]) for s in n.output_shapes]) for n in l._inbound_nodes]) for l in layers])
    counts_outputs = []
    for l in model.layers:
        shapes = []
        for n in l._inbound_nodes:
            if type(n.output_shapes) is list:
                shapes.extend(n.output_shapes)
            else:
                shapes.append(n.output_shapes)
        counts_outputs.append(np.sum([np.prod(s[1:]) for s in shapes]))
    shapes_mem_count += np.sum(counts_outputs)

    trainable_count = np.sum([np.prod(p.shape) for p in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(p.shape) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = batch_size * number_size * (shapes_mem_count + trainable_count + non_trainable_count)

    for s in ['Byte', 'KB', 'MB', 'GB', 'TB']:
        if total_memory > 1024:
            total_memory /= 1024
        else:
            break
    print('memory usage %14.2f %s' % (total_memory, s))


def count_parameters(model):
    trainable_count = int(np.sum([np.prod(p.shape) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([np.prod(p.shape) for p in model.non_trainable_weights]))

    print('trainable     {:>16,d}'.format(trainable_count))
    print('non-trainable {:>16,d}'.format(non_trainable_count))

    return trainable_count + non_trainable_count


def plot_parameter_statistic(model,
                             layer_types=['Dense', 'Conv1D', 'Conv2D', 'Conv3D', 'Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose', 'DepthwiseConv2D'],
                             trainable=True, non_trainable=True, outputs=False, channels=False):
    layer_types = [l.__name__ if type(l) == type else l for l in layer_types]
    layers = get_layers(model)
    layers = [l for l in layers if l.__class__.__name__ in layer_types]
    names = [l.name for l in layers]
    y = range(len(names))

    plt.figure(figsize=(12, 0.2+len(y)/4))

    offset = np.zeros(len(layers), dtype=int)
    legend = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if trainable:
        counts_trainable = [np.sum([np.prod(p.shape) for p in l.trainable_weights]) for l in layers]
        plt.barh(y, counts_trainable, align='center', color=colors[0])
        offset += np.array(counts_trainable, dtype=int)
        legend.append('trainable')
    if non_trainable:
        counts_non_trainable = [np.sum([np.prod(p.shape) for p in l.non_trainable_weights]) for l in layers]
        plt.barh(y, counts_non_trainable, align='center', color=colors[1],  left=offset)
        offset += np.array(counts_non_trainable, dtype=int)
        legend.append('non-trainable')
    if outputs:
        #counts_outputs = [np.sum([np.sum([np.prod(s[1:]) for s in n.output_shapes]) for n in l._inbound_nodes]) for l in layers]
        counts_outputs = []
        for l in layers:
            shapes = []
            for n in l._inbound_nodes:
                if type(n.output_shapes) == list:
                    shapes.extend(n.output_shapes)
                else:
                    shapes.append(n.output_shapes)
            counts_outputs.append(np.sum([np.prod(s[1:]) for s in shapes]))
        plt.barh(y, counts_outputs, align='center', color=colors[2], left=offset)
        offset += np.array(counts_outputs, dtype=int)
        legend.append('outputs')
    if channels:
        counts_channels = [l.output_shape[-1] for l in layers]
        plt.barh(y, counts_channels, align='center', color=colors[3], left=offset)
        offset += np.array(counts_channels, dtype=int)
        legend.append('channels')

    plt.grid()
    plt.yticks(y, names)
    plt.ylim(y[0]-1, y[-1]+1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.legend(legend)
    plt.show()


def plot_kernels(model, distribution=False):
    # plots kernel mean and std for all layers in a model

    layer_names = []
    w = []
    for l in model.layers:
        if hasattr(l, 'kernel'):
            layer_names.append(l.name)
            w.append(np.array(l.kernel).ravel())
    num_layers = len(layer_names)

    if distribution:
        plt.figure(figsize=(6, 1.4*num_layers))
        for i in range(num_layers):
            plt.subplot(num_layers, 1, i+1)
            plt.hist(w[i], bins=100)
            plt.title(layer_names[i])
        plt.tight_layout()
        plt.show()
    else:
        y_mean, y_std = [np.mean(a) for a in w], [np.std(a) for a in w]
        x = np.arange(num_layers)

        plt.figure(figsize=(12, 0.4+0.3*num_layers))
        plt.errorbar(y_mean, x, xerr=y_std, fmt='o')
        plt.yticks(x, layer_names, rotation=0)
        plt.grid(True)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.show()


def plot_activations(model, batch_size=32, distribution=False, ignoere_zeros=False):
    # plots activation mean and std for all layers in a model

    #outputs = [l.output for l in model.layers]
    outputs = [l.output[0] if type(l.output) is list else l.output for l in model.layers]
    tmp_model = Model(model.input, outputs)
    layer_names = [l.name for l in model.layers]
    input_shape = model.input_shape[1:]
    num_layers = len(layer_names)

    if type(model.input_shape) is tuple:
        #x = np.float32(np.random.uniform(-1,1, size=[batch_size, *model.input_shape[1:]]))
        x = np.float32(np.clip(np.random.normal(size=[batch_size, *model.input_shape[1:]]), -3, 3))
    else:
        x = [np.float32(np.clip(np.random.normal(size=[batch_size, *s[1:]]), -3, 3)) for s in model.input_shape]

    y = tmp_model(x)
    y = [np.array(a).flatten() for a in y]

    if ignoere_zeros:
        y = [a[a!=0.0] for a in y]

    if distribution:
        plt.figure(figsize=(6, 1.4*num_layers))
        for i in range(num_layers):
            plt.subplot(num_layers, 1, i+1)
            plt.hist(y[i], bins=100)
            plt.title(layer_names[i])
        plt.tight_layout()
        plt.show()
    else:
        y_mean, y_std = [np.mean(a) for a in y], [np.std(a) for a in y]
        x = np.arange(num_layers)

        plt.figure(figsize=(6, 0.4+0.2*num_layers))
        plt.errorbar(y_mean, x, xerr=y_std, fmt='o')
        plt.yticks(x, layer_names, rotation=0)
        plt.grid(True)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.show()


def plot_activation_with_mask(model, sparsity=0.5, batch_size=32):
    # plots mean and std for layers in PartiaConv model

    conv_layers = [ l for l in model.layers if l.__class__.__name__ in ['PartialConv2D', 'PartialDepthwiseConv2D', 'Lambda'] ]

    input_shape = model.input_shape[0][1:]
    outputs_x = [model.layers[0].output] + [l.output[0] for l in conv_layers] + [model.output[0]]
    outputs_m = [model.layers[1].output] + [l.output[1] for l in conv_layers] + [model.output[1]]
    layer_names = ['input'] + [l.name for l in conv_layers] + ['output']
    num_layers = len(layer_names)

    tmp_model = Model(model.input, outputs_x+outputs_m)

    #x = np.float32(np.random.uniform(-1,1, size=[batch_size, *input_shape]))
    x = np.float32(np.clip(np.random.normal(size=[batch_size, *input_shape]), -3, 3))
    m = np.float32(np.random.binomial(1, 1-sparsity, size=[batch_size, *input_shape]))
    y = tmp_model([x,m])

    y_x, y_m = y[:num_layers], y[num_layers:]
    y_xm = [y_x[i]*y_m[i] for i in range(num_layers)]

    y_x_mean, y_x_std = [np.mean(a) for a in y_x], [np.std(a) for a in y_x]
    y_m_mean, y_m_std = [np.mean(a) for a in y_m], [np.std(a) for a in y_m]
    y_xm_mean, y_xm_std = [np.mean(a) for a in y_xm], [np.std(a) for a in y_xm]

    x = np.arange(num_layers)

    plt.figure(figsize=(16, 0.8+0.2*num_layers))
    plt.subplot(131); plt.title('x')
    plt.errorbar(y_x_mean, x, xerr=y_x_std, fmt='o')
    plt.yticks(x, layer_names, rotation=0)
    plt.grid(True)
    ax = plt.gca()
    ax.invert_yaxis()

    plt.subplot(132); plt.title('xm')
    plt.errorbar(y_xm_mean, x, xerr=y_xm_std, fmt='o')
    plt.yticks(x, [])
    plt.grid(True)
    ax = plt.gca()
    ax.invert_yaxis()

    plt.subplot(133); plt.title('m')
    plt.errorbar(y_m_mean, x, xerr=y_m_std, fmt='o')
    plt.yticks(x, [])
    plt.grid(True)
    ax = plt.gca()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_weights_over_epochs(weight_dir):

    if isinstance(weight_dir, str):
        file_names = sorted(glob(f"{weight_dir}/*.weights.h5"))

    stats = { 'name': [], 'mean': [], 'std': [] }

    def visit_group(name, obj):
        if isinstance(obj, h5py.Group) and name.endswith("/vars"):
            name = obj.attrs['name']
            if name.startswith('conv'):
                w = np.asarray(obj['0'])
                if name not in stats['name']:
                    stats['name'].append(name)
                    stats['mean'].append([])
                    stats['std'].append([])
                idx = stats['name'].index(name)
                stats['mean'][idx].append(np.mean(w))
                stats['std'][idx].append(np.std(w))

    for n in file_names:
        with h5py.File(n, 'r') as f:
            f.visititems(visit_group)

    stats['mean'] = np.asarray(stats['mean'])
    stats['std'] = np.asarray(stats['std'])
    #return stats

    layer_names = stats['name']
    num_layers = len(layer_names)
    num_epochs = stats['mean'].shape[-1]
    x = np.arange(num_layers)

    plt.figure(figsize=(6, 0.4+0.12*num_epochs*num_layers))
    for i in range(num_epochs):
        y_mean = stats['mean'][:,i]
        y_std = stats['std'][:,i]
        plt.errorbar(y_mean, x+i/(num_epochs+1), xerr=y_std, fmt='|', label=str(i), alpha=0.9, markersize=7, capsize=3)
    plt.yticks(x, layer_names, rotation=0)
    plt.ylim(-0.5, num_layers+0.5)
    plt.grid(True)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()


def calc_receptive_field(model, layer_name, verbose=False):
    """Calculate the receptive field related to a certain layer.
    
    # Arguments
        model: Keras model.
        layer_name: Name of the layer.
    
    # Return
        rf: Receptive field (w, h).
        es: Effictive stides in the input image.
        offset: Center of the receptive field associated with the first unit (x, y).
    """
    # TODO...
    
    fstr = '%-20s %-16s %-10s %-10s %-10s %-16s %-10s %-16s'
    if verbose:
        print(fstr % ('name', 'type', 'kernel', 'stride', 'dilation', 'receptive field', 'offset', 'effective stride'))
    l = model.get_layer(layer_name)
    rf = np.ones(2)
    es = np.ones(2)
    offset = np.zeros(2)
    
    while True:
        layer_type = l.__class__.__name__
        k, s, d = (1,1), (1,1), (1,1)
        p = 'same'
        if layer_type in ['Conv2D']:
            k = l.kernel_size
            d = l.dilation_rate
            s = l.strides
            p = l.padding
        elif layer_type in ['MaxPooling2D', 'AveragePooling2D']:
            k = l.pool_size
            s = l.strides
            p = l.padding
        elif layer_type in ['ZeroPadding2D']:
            p = l.padding
        elif layer_type in ['InputLayer', 'Activation', 'BatchNormalization']:
            pass
        else:
            print('unknown layer type %s %s' % (l.name, layer_type))
        
        k = np.array(k)
        s = np.array(s)
        d = np.array(d)
        
        ek = k + (k-1)*(d-1) # effective kernel size
        rf = rf * s + (ek-s)
        es = es * s
        
        if p == 'valid':
            offset += ek/2
            print(ek/2, offset)
        if type(p) == tuple:
            offset -= [p[0][0], p[1][0]]
            print([p[0][0], p[1][0]], offset)
        
        rf = rf.astype(int)
        es = es.astype(int)
        #offset = offset.astype(int)
        if verbose:
            print(fstr % (l.name, l.__class__.__name__, k, s, d, rf, offset, es))
        
        if layer_type == 'InputLayer':
            break
        
        input_name = l.input.name.split('/')[0]
        input_name = input_name.split(':')[0]
        l = model.get_layer(input_name)
    
    return rf, es, offset
