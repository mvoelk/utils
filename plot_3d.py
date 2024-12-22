"""
SPDX-License-Identifier: MIT
Copyright © 2015 - 2024 Markus Völk
Code was taken from https://github.com/mvoelk/utils
"""


import numpy as np
import matplotlib.pyplot as plt

from utils.geometry import trafo, rotx, roty, rotz, rot2quat

from pythreejs import Object3D, AxesHelper, BufferAttribute, Points, Mesh, Shape, Group, LineSegments2
from pythreejs import BufferGeometry, BoxGeometry, ShapeGeometry, LineSegmentsGeometry, CylinderGeometry
from pythreejs import LineMaterial, PointsMaterial, MeshBasicMaterial, MeshPhongMaterial
from pythreejs import Renderer, Scene, PerspectiveCamera, DirectionalLight, AmbientLight, GridHelper, OrbitControls


astuple = lambda M: tuple(np.round(np.reshape(M, (-1,)), 6))

random_color = lambda : '#%02x%02x%02x'%tuple(np.random.randint(0,256, size=3))


def new_arrow(l=0.1, r=0.003, r2=0.006, l2=0.012, color='#ff0000', segments=32):
    material = MeshBasicMaterial(color=color)

    shaft_geometry = CylinderGeometry(r, r, l-l2, segments)
    shaft = Mesh(shaft_geometry, material)
    shaft.position = (0,(l-l2)/2,0)

    head_geometry = CylinderGeometry(0, r2, l2, segments)
    head = Mesh(head_geometry, material)
    head.position = (0,l-l2/2,0)

    arrow = Object3D()
    arrow.add(shaft)
    arrow.add(head)

    return arrow

def new_axes(T=None, p=np.zeros(3), R=np.eye(3), l=0.1, r=0.003, segments=16):

    if T is not None:
        p, R = T[:3,3], T[:3,:3]

    arrow_x = new_arrow(l, r, 2*r, 4*r, color='#ff0000', segments=segments)
    arrow_y = new_arrow(l, r, 2*r, 4*r, color='#00ff00', segments=segments)
    arrow_z = new_arrow(l, r, 2*r, 4*r, color='#0000ff', segments=segments)

    #arrow_x.rotation = (0,0,-np.pi/2, 'XYZ')
    #arrow_y.rotation = (0,0,0, 'XYZ')
    #arrow_z.rotation = (np.pi/2,0,0, 'XYZ')
    arrow_x.quaternion = astuple(rot2quat(rotz(-np.pi/2), True))
    arrow_y.quaternion = astuple(rot2quat(roty(0), True))
    arrow_z.quaternion = astuple(rot2quat(rotx(np.pi/2), True))

    axes = Object3D()
    axes.add([arrow_x, arrow_y, arrow_z])

    axes.position = astuple(p)
    axes.quaternion = astuple(rot2quat(R, True))

    return axes

def new_cloud(pts, colors=None, color=None, point_size=0.001):
    """
    # Arguments
        pts: float, shape (n, 3)
        colors: uint8, shape (n, 3)
        color: str, '#ff0000'
    """
    position = BufferAttribute(array=np.float32(pts))
    if colors is not None:
        color = BufferAttribute(array=np.float32(colors))
        geometry = BufferGeometry(attributes={'position': position, 'color': color})
        material = PointsMaterial(vertexColors='VertexColors', size=point_size)
    elif color is not None:
        geometry = BufferGeometry(attributes={'position': position})
        material = PointsMaterial(color=color, size=point_size)
    else:
        z = pts[:,2]
        z_min, z_max = np.min(z), np.max(z)
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)
        cmap = plt.get_cmap('viridis')
        colors = cmap(z_norm)[:,:3]
        color = BufferAttribute(array=np.float32(colors))
        geometry = BufferGeometry(attributes={'position': position, 'color': color})
        material = PointsMaterial(vertexColors='VertexColors', size=point_size)

    return Points(geometry=geometry, material=material)


def new_frame(T=None, p=np.zeros(3), R=np.eye(3), frame_size=0.05):
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    frame = AxesHelper(frame_size)
    frame.position = astuple(p)
    frame.quaternion = astuple(rot2quat(R, True))
    return frame


def new_bounding_box(box_size, T=None, p=np.zeros(3), R=np.eye(3), d=0.0005, color='#ff0000', segments=8):
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    
    s, l, h = box_size
    
    pos = (-s/2, -l/2, -h/2)
    
    material = MeshBasicMaterial(color=color)
    
    box = Object3D()
    
    geometry_h = CylinderGeometry(d, d, h, segments)
    mesh_h = Object3D()
    for pm in [(0,-h/2,0), (s,-h/2,0), (0,-h/2,l), (s,-h/2,l)]:
        mesh = Mesh(geometry_h, material)
        mesh.position = pm
        mesh_h.add(mesh)
    #mesh_h.rotation = (-np.pi/2,0,0, 'XYZ')
    mesh_h.quaternion = astuple(rot2quat(rotx(-np.pi/2), True))
    mesh_h.position = pos
    box.add(mesh_h)

    geometry_s = CylinderGeometry(d, d, s, segments)
    mesh_s = Object3D()
    for pm in [(0,s/2,0), (-l,s/2,0), (0,s/2,h), (-l,s/2,h)]:
        mesh = Mesh(geometry_s, material)
        mesh.position = pm
        mesh_s.add(mesh)
    #mesh_s.rotation = (0,0,-np.pi/2, 'XYZ')
    mesh_s.quaternion = astuple(rot2quat(rotz(-np.pi/2), True))
    mesh_s.position = pos
    box.add(mesh_s)
    
    geometry_l = CylinderGeometry(d, d, l, segments)
    mesh_l = Object3D()
    for pm in [(0,l/2,0), (s,l/2,0), (0,l/2,h), (s,l/2,h)]:
        mesh = Mesh(geometry_l, material)
        mesh.position = pm
        mesh_l.add(mesh)
    #mesh_l.rotation = (0,0,0, 'XYZ')
    #mesh_l.quaternion = astuple(rot2quat(rotx(0), True))
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

def new_range(xyz_range, color='#00ff00', opacity=0.25):
    x_range, y_range, z_range = xyz_range
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    geometry = BoxGeometry(width=(x_max-x_min), height=(y_max-y_min), depth=(z_max-z_min), widthSegments=1, heightSegments=1, depthSegments=1)
    material = MeshPhongMaterial(color=color, opacity=opacity, transparent=True, wireframe=False)
    range_box = Mesh(geometry=geometry, material=material)
    range_box.position = ((x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2)
    range_box.quaternion = (0,0,0,1)
    return range_box


def update_pose(frame, T=None, p=np.zeros(3), R=np.eye(3)):
    if T is not None:
        p, R = T[:3,3], T[:3,:3]
    frame.position = astuple(p)
    frame.quaternion = astuple(rot2quat(R, True))
    return frame



def show_cloud(xyz, rgb=None, Tcw=None, 
               frames_in_world=[], frames_in_camera=[], ranges_in_world=[], ranges_in_camera=[],
               width=800, height=600, cloud_subsampling=1, point_size=0.008, frame_size=0.1):
    """Displays a point cloud in jupyter notebook

    # Arguments
        xyz: float, shape (..., 3)
        rgb: uint8, shape (..., 3)
        width: width of the notebook widget
        height: height of the notebook widget
        Tcw: camera frame, homogeneous transformation, shape (4, 4)
        frames_in_world: homogeneous transformations, shape (n, 4, 4)
        frames_in_camera: homogeneous transformations, shape (n, 4, 4)
        ranges_in_world: [[[x_min, x_max], [y_min, y_max], [z_min, z_max]], ...], shape (n, 3, 2)
        ranges_in_camera: [[[x_min, x_max], [y_min, y_max], [z_min, z_max]], ...], shape (n, 3, 2)
    """

    frames_in_world = np.reshape(frames_in_world, (-1,4,4))
    frames_in_camera = np.reshape(frames_in_camera, (-1,4,4))
    ranges_in_world = np.reshape(ranges_in_world, (-1,3,2))
    ranges_in_camera = np.reshape(ranges_in_camera, (-1,3,2))

    Tw = trafo(R=rotx(-np.pi/2))

    if Tcw is None:
        Tcw = np.eye(4)
        Tw = trafo(R=rotz(np.pi))
    
    n = cloud_subsampling
    xyz = xyz.reshape(-1,3)[::n,:]
    if rgb is not None:
        rgb = rgb.reshape(-1,3)[::n,:] / 255
    cloud = new_cloud(xyz, rgb, point_size=point_size)
    
    cloud_frame = new_frame(frame_size=frame_size)
    cloud_frame.add(cloud)
    
    cam_frame = new_frame(Tcw, frame_size=frame_size)
    cam_frame.add(cloud_frame)
    
    for T in frames_in_camera:
        #cam_frame.add(new_frame(T, frame_size=frame_size))
        cam_frame.add(new_axes(T, l=frame_size, r=0.03*frame_size))
    
    for xyz_range in ranges_in_camera:
        cam_frame.add(new_range(xyz_range))

    world_frame = new_frame(Tw, frame_size=frame_size)
    world_frame.add(cam_frame)

    for T in frames_in_world:
        #world_frame.add(new_frame(T, frame_size=frame_size))
        world_frame.add(new_axes(T, l=frame_size, r=0.03*frame_size))
    
    for xyz_range in ranges_in_world:
        world_frame.add(new_range(xyz_range))

    cam = PerspectiveCamera(up=[0,1,0], near=0.1, far=20.0,
        children=[
            DirectionalLight(color='white', position=(3,5,1), intensity=0.5)
        ])
    
    scene = Scene(
        children=[cam, world_frame,
            AmbientLight(color='#777777'),
            GridHelper(size=4, divisions=8, colorCenterLine='#888888', colorGrid='#444444'),
        ])

    Tc = Tw@Tcw
    #xyz_ = np.reshape(xyz, (-1,3))
    #xyz_mean = np.mean(xyz_[xyz_[:,2]>1e-5], axis=0)
    #Tt = Tc @ trafo(t=xyz_mean)
    Tt = Tc @ trafo(t=(0,0,2))
    
    target = astuple(Tt[:3,3])
    orbit = OrbitControls(controlling=cam, target=target)
    
    renderer = Renderer(camera=cam, scene=scene, controls=[orbit], width=width, height=height)

    cam.position = astuple(Tc[:3,3])
    cam.quaternion = astuple(rot2quat(Tc[:3,:3], False)*[-1,-1,1,1]) # but why?
    
    return renderer
