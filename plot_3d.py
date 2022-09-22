
import numpy as np

from utils.geometry import rot2quat, euler2rot2

from pythreejs import Object3D, AxesHelper, BufferAttribute, Points, Mesh, Shape, Group, LineSegments2
from pythreejs import BufferGeometry, BoxGeometry, ShapeGeometry, LineSegmentsGeometry, CylinderGeometry
from pythreejs import LineMaterial, PointsMaterial, MeshBasicMaterial, MeshPhongMaterial


astuple = lambda M: tuple(np.round(np.reshape(M, (-1,)), 6))


def new_arrow(l=0.1, r=0.003, r2=0.006, l2=0.012, color='#ff0000'):
    material = MeshBasicMaterial(color=color)

    shaft_geometry = CylinderGeometry(r, r, l-l2, 32)
    shaft = Mesh(shaft_geometry, material)
    shaft.position = (0,(l-l2)/2,0)

    head_geometry = CylinderGeometry(0, r2, l2, 32)
    head = Mesh(head_geometry, material)
    head.position = (0,l-l2/2,0)

    arrow = Object3D()
    arrow.add(shaft)
    arrow.add(head)

    return arrow

def new_axes(T=None, p=np.zeros(3), R=np.eye(3), l=0.1, r=0.003):
    
    if T is not None:
        p, R = T[:3,3], T[:3,:3]

    arrow_x = new_arrow(l, r, 2*r, 4*r, color='#ff0000')
    arrow_y = new_arrow(l, r, 2*r, 4*r, color='#00ff00')
    arrow_z = new_arrow(l, r, 2*r, 4*r, color='#0000ff')

    arrow_x.rotation = (0,0,-np.pi/2, 'XYZ')
    arrow_y.rotation = (0,0,0, 'XYZ')
    arrow_z.rotation = (np.pi/2,0,0, 'XYZ')

    axes = Object3D()
    axes.add([arrow_x, arrow_y, arrow_z])
    
    axes.position = astuple(p)
    axes.quaternion = astuple(rot2quat(R, True))

    return axes


def new_cloud(pts, colors=None, color='#ff0000', point_size=0.001):
    position = BufferAttribute(array=np.float32(pts))
    if colors is not None:
        color = BufferAttribute(array=np.float32(colors))
        geometry = BufferGeometry(attributes={'position': position, 'color': color})
        material = PointsMaterial(vertexColors='VertexColors', size=point_size)
    else:
        geometry = BufferGeometry(attributes={'position': position})
        material = PointsMaterial(color=color, size=point_size)
    return Points(geometry=geometry, material=material)


def new_frame(T=None, p=np.zeros(3), R=np.eye(3), frame_size=0.05):
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    frame = AxesHelper(frame_size)
    frame.position = astuple(p)
    frame.quaternion = astuple(rot2quat(R, True))
    return frame


def new_bounding_box(box_size, T=None, p=np.zeros(3), R=np.eye(3), d=0.0005, color='#ff0000'):
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    
    s, l, h = box_size
    
    pos = (-s/2, -l/2, -h/2)
    
    material = MeshBasicMaterial(color=color)
    
    box = Object3D()
    
    geometry_h = CylinderGeometry(d, d, h, 32)
    mesh_h = Object3D()
    for pm in [(0,-h/2,0), (s,-h/2,0), (0,-h/2,l), (s,-h/2,l)]:
        mesh = Mesh(geometry_h, material)
        mesh.position = pm
        mesh_h.add(mesh)
    mesh_h.rotation = (-np.pi/2,0,0, 'XYZ')
    mesh_h.position = pos
    box.add(mesh_h)

    geometry_s = CylinderGeometry(d, d, s, 32)
    mesh_s = Object3D()
    for pm in [(0,s/2,0), (-l,s/2,0), (0,s/2,h), (-l,s/2,h)]:
        mesh = Mesh(geometry_s, material)
        mesh.position = pm
        mesh_s.add(mesh)
    mesh_s.rotation = (0,0,-np.pi/2, 'XYZ')
    mesh_s.position = pos
    box.add(mesh_s)
    
    geometry_l = CylinderGeometry(d, d, l, 32)
    mesh_l = Object3D()
    for pm in [(0,l/2,0), (s,l/2,0), (0,l/2,h), (s,l/2,h)]:
        mesh = Mesh(geometry_l, material)
        mesh.position = pm
        mesh_l.add(mesh)
    #mesh_l.rotation = (0,0,0, 'XYZ')
    mesh_l.position = pos
    box.add(mesh_l)
    
    box.position = astuple(p)
    box.quaternion = astuple(rot2quat(R, True))
    
    return box

def new_box(box_size, T=None, p=np.zeros(3), R=np.eye(3), color='#ff0000', opacity=0.5):
    w,h,d = box_size
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    geometry = BoxGeometry(width=w, height=h, depth=d, widthSegments=1, heightSegments=1, depthSegments=1)
    material = MeshPhongMaterial(color=color, opacity=opacity, transparent=True, wireframe=False)
    box = Mesh(geometry=geometry, material=material)
    box.position = astuple(p)
    box.quaternion = astuple(rot2quat(R, True))
    return box


def update_pose(frame, T=None, p=np.zeros(3), R=np.eye(3)):
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    frame.position = astuple(p)
    frame.quaternion = astuple(rot2quat(R, True))
    return frame

