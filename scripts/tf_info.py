#!/usr/bin/env python3

import os, platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

import tensorflow as tf

print('versions')
print(' '*4+'%-26s %s'%('python', platform.python_version()))
print(' '*4+'%-26s %s'%('tensorflow', tf.__version__))


print('build info')
for k, v in tf.sysconfig.get_build_info().items():
    print(' '*4+'%-26s %s'%(k,v))

print('devices')
for d in tf.config.get_visible_devices():
    print(' '*4+'%s'%(d.name))
    for k, v in tf.config.experimental.get_device_details(d).items():
        print(' '*8+'%-22s %s'%(k,v))

