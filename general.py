
import re, random


def print_json_tree(json):
    def recurse(d, o):
        pad = '  '
        if isinstance(o, dict):
            for k in o.keys():
                print(pad*d + k)
                recurse(d+1, o[k])
        if isinstance(o, list) and len(o) and isinstance(o[0], dict):
            print(pad*d + '...')
            recurse(d+1, o[0])
    recurse(0, json)

def find_json_key(json, key, pprint=True):
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
    recurse([], json)
    if not pprint:
        return found


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


