"""
SPDX-License-Identifier: MIT
Copyright © 2017 - 2025 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""

import re, random, time
import cProfile


class Object():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class measure:
    """Context manager for measuring execution time

    Example:
        with measure('Foobar'):
            time.sleep(1)
    """

    def __init__(self, name='Execution'):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.readout = f'{self.name} took {self.time:.3f} seconds'
        print(self.readout)

class profile:
    """Context manager for profiling

    Example:
        with profile():
            time.sleep(1)
    """

    def __init__(self, sort='time'):
        self.sort = sort

    def __enter__(self):
        self.profile = cProfile.Profile()
        self.profile.enable()
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.profile.disable()
        self.profile.print_stats(sort=self.sort)


def print_json_tree(json_data):
    def recurse(d, o):
        pad = '  '
        if isinstance(o, dict):
            for k in o.keys():
                print(pad*d + k)
                recurse(d+1, o[k])
        if isinstance(o, list) and len(o) and isinstance(o[0], dict):
            print(pad*d + '...')
            recurse(d+1, o[0])
    recurse(0, json_data)

def find_json_key(json_data, key, pprint=True):
    # key can be a regex
    if isinstance(key, str):
        key = re.compile(key)
    found = []
    def recurse(p, o):
        if isinstance(o, dict):
            for k in o.keys():
                #if k == key:
                if re.match(key, k):
                    found.append(p+[k])
                    if pprint:
                        #print('%-6s %s' % (type(o[k]).__name__, ' > '.join(p+[k])))
                        print('%-6s %s' % (type(o[k]).__name__, ''.join(['[\'%s\']'%(s) for s in p+[k]])))
                recurse(p+[k], o[k])
        if isinstance(o, list) and len(o) and isinstance(o[0], dict):
            recurse(p+['...'], o[0])
    recurse([], json_data)
    if not pprint:
        return found

def print_h5_info(file_path):
    import h5py

    def print_attrs(name, obj):
        s = '%-8s %-70s' % (obj.__class__.__name__, name)
        if isinstance(obj, h5py.Group):
            s += ' '.join(['%s = %s'%(k,v) for k,v in obj.attrs.items()])
        elif isinstance(obj, h5py.Dataset):
            s += str(obj.shape)
        print(s)

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)


def random_derangement(n):
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)


