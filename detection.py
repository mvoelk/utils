"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyclipper


def metrics(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fmeasure = 2 * (precision*recall) / (precision+recall + eps)
    return precision, recall, fmeasure


def rot_matrix(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[c, -s],[s, c]])


def draw_bbxes(boxes, labels, classes, show_labels=True, box_format='xywh', normalized=False):
    boxes = np.float32(boxes)
    num_classes = len(classes)
    classes_lower = [s.lower() for s in classes]
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes+1)).tolist()

    ax = plt.gca()
    im = plt.gci()
    w, h = im.get_size()

    for i in range(len(boxes)):
        box = boxes[i]
        class_idx = int(labels[i])
        color = colors[class_idx]
        if box_format == 'xywh': # opencv
            xmin, ymin, w, h = box
            xmax, ymax = xmin + w, ymin + h
        elif box_format == 'xyxy':
            xmin, ymin, xmax, ymax = box
        if box_format == 'polygon':
            xy = box.reshape((-1,2))
            is_polygon = True
        else:
            xy = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            is_polygon = False
        ax.add_patch(plt.Polygon(xy, fill=False, edgecolor=color, linewidth=1))
        if show_labels:
            label_name = classes[class_idx]
            if is_polygon:
                angle = np.arctan((xy[1,0]-xy[0,0])/(xy[1,1]-xy[0,1]+eps))
                if angle < 0:
                    angle += np.pi
                angle = angle/np.pi*180-90
            else:
                angle = 0
            ax.text(xy[0,0], xy[0,1], label_name, bbox={'facecolor':color, 'alpha':0.5}, rotation=angle)


def draw_corners(corners):
    # draws enumerated points form cv2.findChessboardCorners etc.
    colors = cm.rainbow(np.linspace(0, 1, len(corners)))
    for i in range(len(corners)):
        xy = corners[i,0,:]
        plt.plot(xy[0], xy[1], 'o', color=colors[i], markersize=4)
        plt.text(xy[0], xy[1], i, color='c')
        
def draw_frame(T, K, scale=0.15, style=''):
    # draws projected homogeneous transformation in image plane
    r1, r2, r3, p0 = T[:3,:].T
    p1, p2, p3 = p0+r1*scale, p0+r2*scale, p0+r3*scale
    p1, p2, p3, p0 = K@p1/p1[2], K@p2/p2[2], K@p3/p3[2], K@p0/p0[2]

    plt.plot([p0[0]], [p0[1]], 'ro', markersize=4)
    plt.plot([p0[0],p1[0]], [p0[1],p1[1]], '-r'+style, markersize=4)
    plt.plot([p0[0],p2[0]], [p0[1],p2[1]], '-g'+style, markersize=4)
    plt.plot([p0[0],p3[0]], [p0[1],p3[1]], '-b'+style, markersize=4)

def draw_frame_(T=None, p=np.zeros(3), R=np.eye(3), text='', scale=25, style=''):
    # draws a rotation matrix in the image plane
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    Rv = R * scale
    plt.plot([p[0], p[0]+Rv[0,0]], [p[1], p[1]+Rv[1,0]], 'r'+style)
    plt.plot([p[0], p[0]+Rv[0,1]], [p[1], p[1]+Rv[1,1]], 'g'+style)
    plt.plot([p[0], p[0]+Rv[0,2]], [p[1], p[1]+Rv[1,2]], 'b'+style)
    plt.plot(p[0], p[1], 'ro', markersize=max((scale*0.2)**0.6,2))
    d = scale/2
    plt.text(p[0]+d, p[1]+d, text)

def draw_keypoints(xy, marker='o', cmap='tab10'):
    ax = plt.gca()
    cmap = plt.get_cmap(cmap)
    for i in range(min(len(cmap.colors),len(xy))):
        ax.plot(xy[i,0], xy[i,1], marker=marker, color=cmap(i))

def draw_orientation(xy, angle, length=32):
    x, y = xy
    ax = plt.gca()
    ax.plot([x,x+length*np.sin(angle)], [y,y+length*np.cos(angle)], 'r-')
    ax.plot(x, y, 'ro')

def get_neighbors(shape=(6,4)):
    map_h, map_w = shape
    xy_pos = np.asanyarray(np.meshgrid(np.arange(map_w), np.arange(map_h))).reshape(2,-1).T
    xy = np.tile(xy_pos, (1,8))
    xy += np.array([-1,-1, 0,-1, +1,-1,
                    -1, 0,       +1, 0,
                    -1,+1, 0,+1, +1,+1])
    valide = (xy[:,0::2] >= 0) & (xy[:,0::2] < map_w) & (xy[:,1::2] >= 0) & (xy[:,1::2] < map_h)
    idxs = xy[:,1::2] * map_w + xy[:,0::2]
    idxs[np.logical_not(valide)] = -1
    return idxs


def color_id_map(id_map):
    img = np.zeros((*id_map.shape, 3), dtype='float32')
    for i in np.unique(id_map):
        if i == 0:
            c = 0.1 * np.ones(3)
        else:
            c = np.random.random(3)
            a = 0.9
            c = a*c + (1-a)*np.ones_like(c)
        img += np.repeat((id_map==i)[:,:,None], 3, axis=-1) * c
    return img


def crop_random_patch(img, size):
    h, w = img.shape[:2]
    assert h >= size and w >= size
    x1 = np.random.randint(w-size) if w-size > 0 else 0
    y1 = np.random.randint(h-size) if h-size > 0 else 0
    x2, y2 = x1+size, y1+size
    img = img[y1:y2,x1:x2,:]
    return img


def iou_polygon(poly1, poly2):
    # poly1/poly2 shape (-1, 2)
    pc = pyclipper.Pyclipper()
    pc.AddPath(poly1, pyclipper.PT_CLIP, True)
    pc.AddPath(poly2, pyclipper.PT_SUBJECT, True)
    I = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    if len(I) > 0:
        U = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        Ia = pyclipper.Area(I[0])
        Ua = pyclipper.Area(U[0])
        IoU = Ia / Ua
    else:
        IoU = 0.0
    return IoU

def iom_polygon(poly1, poly2):
    # intersection over minimum
    # poly1/poly2 shape (-1, 2)
    pc = pyclipper.Pyclipper()
    pc.AddPath(poly1, pyclipper.PT_CLIP, True)
    pc.AddPath(poly2, pyclipper.PT_SUBJECT, True)
    I = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    if len(I) > 0:
        Ia = pyclipper.Area(I[0])
        Ma = min(abs(pyclipper.Area(poly1)), abs(pyclipper.Area(poly2)))
        IoM = Ia / Ma
    else:
        IoM = 0.0
    return IoM

