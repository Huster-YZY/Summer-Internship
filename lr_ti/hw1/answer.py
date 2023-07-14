import taichi as ti

ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)
use_implicit = True

n = 50
num_iteration = 3
quad_size = 1.0 / n
dt = 8e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e2
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
num_nodes = n * n
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False
F = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
J = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(num_nodes, num_nodes))
A = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(num_nodes, num_nodes))
M = ti.field(dtype=ti.f32, shape=(num_nodes, num_nodes))
new_v = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
b = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
I = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@ti.func
def m2v(x):
    return x[0] * n + x[1]


@ti.func
def v2m(x):
    return (int(x / n), int(x % n))


@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 2 + j // 2) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)


initialize_mesh_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))


@ti.kernel
def update_force():
    for i in ti.grouped(x):
        idx = m2v(i)
        F[idx] = gravity
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                F[idx] += -spring_Y * d * (current_dist / original_dist - 1)


@ti.kernel
def update_velocity():  #for explicit method
    for i in ti.grouped(v):
        idx = m2v(i)
        v[i] += F[idx] * dt


@ti.kernel
def update_Jacobi():
    for i, d in J:
        J[i, d] *= 0.0
        idx_i = v2m(i)
        for offset in ti.static(spring_offsets):
            idx_j = idx_i + offset
            j = m2v(idx_j)
            if d == i or d == j:
                x_ij = x[idx_i] - x[idx_j]
                dir = x_ij.normalized()
                mat = dir.outer_product(dir)
                r = quad_size * float(idx_i - idx_j).norm()
                l = x_ij.norm()
                jx = -(spring_Y / r) * ((1 - r / l) * (I - mat) + mat)
                if d == i:
                    J[i, d] += jx
                else:
                    J[i, d] -= jx


@ti.kernel
def update_Ab():
    for i, j in A:
        A[i, j] = M[i, j] * I - dt * dt * J[i, j]
    for i in b:
        b[i] = v[v2m(i)] + dt * F[i]


@ti.kernel
def collision_detection():
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]


@ti.kernel
def Jacobi_iteration():
    n = num_nodes
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] @ v[v2m(j)]
        new_v[i] = A[i, i].inverse() @ r
    for i in ti.grouped(v):
        v[i] = new_v[m2v(i)]


@ti.kernel
def init_m():
    for i in range(num_nodes):
        M[i, i] = 1.0


def substep():
    update_force()
    update_velocity()
    collision_detection()


def implicit_substep(num_iteration):
    update_force()
    update_Jacobi()
    update_Ab()
    for _ in range(num_iteration):
        Jacobi_iteration()
    collision_detection()


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()
init_m()

# dump to images
# result_dir = "../../video"
# video_manager = ti.tools.VideoManager(output_dir=result_dir,
#                                       framerate=60,
#                                       automatic_build=True)

while window.running:
    if current_t > 50:
        # break
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        if use_implicit:
            implicit_substep(num_iteration)
        else:
            substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center,
                    radius=ball_radius * 0.8,
                    color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    # video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()