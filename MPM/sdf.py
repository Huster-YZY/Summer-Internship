import taichi as ti
import numpy as np
import sys
import time
from tqdm import tqdm
import trimesh
# import mesh_to_sdf as md
import mesh2sdf

# ti.init(arch=ti.cpu)

MAX_VERTEX_NUM = 10  #25000
FACE_NUM = 20  #50000
MAX_FACE_NUM = 3 * FACE_NUM

window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

vertices = ti.Vector.field(3, dtype=float, shape=MAX_VERTEX_NUM)
indices = ti.field(dtype=int, shape=MAX_FACE_NUM)
num_vertices = ti.field(dtype=int, shape=())
rotation_matrix = ti.Matrix.field(3, 3, dtype=float, shape=())

PI = 3.1415926

mesh = None
size = None
level = None
num_grid = None
dx = None

reverse_rotation_matrix = ti.Matrix.field(3, 3, dtype=float, shape=())
reverse_rotation_matrix[None] = ti.Matrix([[1.0, 0.0, 0.0], \
                                           [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0]])
reverse_offset_vector = ti.Vector.field(3, dtype=float, shape=())
position = ti.Vector.field(3, dtype=float, shape=(32, 32, 32))
tmp_sdf = ti.field(dtype=float, shape=(32, 32, 32))
static_sdf = ti.field(dtype=float, shape=(64, 64, 64))
positive_infinity = 99999.99


def load_mesh_fast(filename, n_grid, scale_ratio=1.0):
    global mesh
    global size
    global level
    global num_grid
    global dx

    mesh = trimesh.load(filename, force='mesh')
    size = n_grid * 2
    level = 2 / size
    num_grid = n_grid
    dx = 1.0 / n_grid

    t0 = time.time()

    mesh.vertices *= 2
    mesh.vertices[:, 0] += 0.3
    mesh.vertices[:, 1] += 0.3

    sdf, _ = mesh2sdf.compute(mesh.vertices,
                              mesh.faces,
                              size,
                              fix=True,
                              level=level,
                              return_mesh=True)
    t1 = time.time()
    print('It takes %.4f seconds to process %s' % (t1 - t0, filename))

    static_sdf.from_numpy(sdf)
    sdf = sdf[-n_grid:, -n_grid:, -n_grid:]

    num_vertices[None] = len(mesh.vertices)
    for i, vertex in enumerate(mesh.vertices):
        vertices[i] = ti.Vector(scale_ratio * vertex)
    for i, face in enumerate(mesh.faces):
        for j in range(3):
            indices[3 * i + j] = face[j]

    return sdf


# def gen_point_cloud():
#     global pld
#     pld = md.get_surface_point_cloud(mesh)

# def get_sdf(x):
#     global pld
#     return pld.get_sdf_in_batches(x)


@ti.kernel
def rotate(ang_x: float, ang_y: float, ang_z: float):
    rotation_matrix[None] = ti.math.rotation3d(ang_x, ang_y, ang_z)

    #reverse transform
    tmp = ti.math.rotation3d(-ang_x, -ang_y, -ang_z)
    reverse_rotation_matrix[None] = tmp @ reverse_rotation_matrix[None]
    reverse_offset_vector[None] = tmp @ reverse_offset_vector[None]

    for i in range(num_vertices[None]):
        vertices[i] = rotation_matrix[None] @ vertices[i]


@ti.func
def transform(x: float, y: float, z: float):
    bias = ti.Vector([x, y, z])

    #reverse transform
    reverse_offset_vector[None] -= bias

    #apply transform
    for i in range(num_vertices[None]):
        vertices[i] += bias


def update_sdf():
    sdf, _ = mesh2sdf.compute(vertices.to_numpy()[:num_vertices[None]],
                              mesh.faces,
                              size,
                              fix=True,
                              level=level,
                              return_mesh=True)
    return sdf[-num_grid:, -num_grid:, -num_grid:]


@ti.kernel
def apply_reverse_transform():
    #switch reference frame
    for i in ti.grouped(position):
        position[i] = \
            reverse_rotation_matrix[None] @ position[i] + reverse_offset_vector[None]

    #compute new sdf
    t = num_grid
    for i in ti.grouped(position):
        x, y, z = position[i] * num_grid
        #for debug
        # print(x, y, z)
        if x <= -t or x > t or y <= -t or y > t or z <= -t or z > t:
            tmp_sdf[i] = positive_infinity
        else:
            #can be more precise if you use interpolation
            idx = ti.Vector([x, y, z], dt=int) + 32
            # print(idx)
            tmp_sdf[i] = static_sdf[idx[0], idx[1], idx[2]]


def switch_reference_frame_and_update_sdf(pos):
    position.from_numpy(pos)
    apply_reverse_transform()
    return tmp_sdf.to_numpy()


# result_dir = "../video"
# video_manager = ti.tools.VideoManager(output_dir=result_dir,
#                                       framerate=60,
#                                       automatic_build=True)


def main():
    while window.running:
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)

        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(vertices, indices, two_sided=True)
        angle = 0.05
        rotate(0, 0, angle)
        canvas.scene(scene)
        # video_manager.write_frame(window.get_image_buffer_as_numpy())
        window.show()


if __name__ == "__main__":
    # filename = './model/cow.obj'
    # start_time = time.time()
    # load_mesh_fast(filename)
    # use_time = time.time() - start_time
    # print(f"Load Successfully.\nTime Consumption:{use_time}s")
    # main()
    apply_reverse_transform()
