#!/usr/bin/env python3

import sys, os, platform

print('%-26s %s'%('executable', sys.executable))
print('%-26s %s'%('version', platform.python_version()))

print('path')
for p in sys.path:
    print(' '*4 + p)

print('current working directory')
print(' '*4 + os.getcwd())

print('environment variables')
for k in os.environ:
    print(' '*4 + '%s=%s' % (k, os.environ[k]))
