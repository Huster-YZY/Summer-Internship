EXPORT = False
if EXPORT:
    export_file = "./frames/mpm3d.ply"
else:
    export_file = ""

import numpy as np
from plyImporter import PlyImporter

import taichi as ti

ti.init(arch=ti.metal)
from sdf import load_mesh_fast, gen_point_cloud, get_sdf, vertices, indices

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 32, 25, 4e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 25, 8e-5

# ply3 = PlyImporter("/Users/YZY/g201/MPM/frames/mpm3d_000000.ply")
# ply3 = PlyImporter("/Users/YZY/g201/MPM/model/bunny.ply")

n_particles = n_grid**dim // 2**(dim - 1)
# n_particles = ply3.get_count()

dx = 1 / n_grid
inv_dx = n_grid
p_rho = 1
# p_vol = (dx * 0.5)**2
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E, nu = 400, 0.2
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
# F_J = ti.field(float, n_particles)
F = ti.Matrix.field(dim, dim, float, n_particles)

F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)
pos = ti.Vector.field(dim, dtype=float, shape=(n_grid, ) * dim)
SDF = ti.field(float, (n_grid, ) * dim)
CDF = ti.Vector.field(dim, dtype=float, shape=n_grid**dim)

grid_pos = ti.Vector.field(dim, dtype=float, shape=n_grid**dim)
ref_point = ti.Vector([0.3, 0.0, 0.0])
normal = ti.Vector([1.0, 1.0, 1.0]).normalized()

# co_position = ti.Vector.field(dim, float, 1)
# co_v = ti.Vector.field(dim, float, 1)
fe = 0.0  #friction coefficient

neighbour = (3, ) * dim
ORIANGE = (0.9294117647058824, 0.3333333333333333, 0.23137254901960785)
PI = 3.1415926


@ti.func
def P2G():
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # stress = -dt * 4 * E * p_vol * (F_J[p] - 1) / dx**2
        # affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]
        I = ti.Matrix.identity(float, dim)
        F[p] = (I + dt * F_C[p]) @ F[p]
        J = 1.0
        _, sig, _ = ti.svd(F[p])
        for i in ti.static(range(dim)):
            J *= sig[i, i]

        #Neo-Hookean
        stress = mu * (F[p] @ F[p].transpose() - I) + I * la * ti.log(J)

        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

        #PIC
        # affine = stress
        #APIC
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base +
                     offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass


@ti.func
def pp(x):
    print(x)


@ti.func
def Boundary():
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
            F_grid_v[I][1] -= dt * gravity

        #Central Ball Collision
        # grid_pos = I * dx
        # d = grid_pos - co_position[0]
        # if d.norm_sqr() <= 0.01:
        #     n = d.normalized()
        #     v_ref = F_grid_v[I] - co_v[0]
        #     vn = v_ref.dot(n)
        #     if vn < 0.0:
        #         vt = v_ref - vn * n
        #         v_ref = vt + fe * vn * vt.normalized()
        #         F_grid_v[I] = v_ref + co_v[0]

        #SDF collision solver
        if SDF[I] < 0.0:
            i, j, k = I[0], I[1], I[2]
            dx = (SDF[i + 1, j, k] - SDF[i - 1, j, k]) * 0.5 * inv_dx
            dy = (SDF[i, j + 1, k] - SDF[i, j - 1, k]) * 0.5 * inv_dx
            dz = (SDF[i, j, k + 1] - SDF[i, j, k - 1]) * 0.5 * inv_dx
            grad = ti.Vector([dx, dy, dz])
            n = grad.normalized()

            #TIPS: Now we assume the object is static
            v_ref = F_grid_v[I]
            vn = v_ref.dot(n)
            if vn < 0.0:
                vt = v_ref - vn * n
                v_ref = vt + fe * vn * vt.normalized()
                F_grid_v[I] = v_ref

        #TODO:add friction between material and the floor
        # Box Constraint
        cond = (I < bound) & (F_grid_v[I] <
                              0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])


@ti.func
def G2P():
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = offset - fx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        # F_J[p] *= 1 + dt * new_C.trace()
        F_C[p] = new_C


@ti.kernel
def substep():
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=n_grid, parallelize=n_grid)
    P2G()
    ti.loop_config(block_dim=n_grid, parallelize=n_grid)
    Boundary()
    ti.loop_config(block_dim=n_grid, parallelize=n_grid)
    G2P()


@ti.kernel
def init():
    # co_position[0] = ti.Vector([0.2, 0.1, 0.2])
    # co_v[0] = ti.Vector([0.0, 0.0, 0.0])
    for i in range(n_particles):
        F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.4 + 0.3
        #!!pay attention to the grid index
        #!!if computed index is negative, then may lead to crush!!!
        F_x[i][0] -= 0.3
        F_x[i][2] -= 0.3
        F_x[i][1] += 0.2
        # F_J[i] = 1
        F[i] = ti.Matrix.identity(float, dim)


def T(a):
    '''
    Project 3d circles on 2d screen
    '''
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5


@ti.kernel
def update_co_position():
    t = co_position[0][1] + dt * co_v[0][1]
    if t >= 1.0 or t < 0.1:
        co_v[0] *= -1.0
    co_position[0] += dt * co_v[0]


@ti.kernel
def init_pos():
    for i in ti.grouped(pos):
        pos[i] = i * dx
        # SDF[i] = (pos[i] - ref_point).dot(normal)


def init_sdf(SignedDistanceField):
    init_pos()
    SDF.from_numpy(SignedDistanceField)

    t = pos.to_numpy().reshape((-1, 3))
    grid_pos.from_numpy(t)

    # dis = SDF.to_numpy().reshape((-1))

    #use mesh_to_sdf compute the SDF
    # gen_point_cloud()
    # dis = get_sdf(t)
    # SDF.from_numpy(dis.reshape(SDF.shape))

    print(
        "************Signed Distance Field has been initialized.************")

    #visualize SDF
    # low = dis.min()
    # high = dis.max()
    # colors = (dis - low) / (high - low)
    # for i, c in enumerate(colors):
    #     CDF[i] = ti.Vector([c, c, c])
    # print("************Color Distance Field has been initialized.************")


result_dir = "../video"
video_manager = ti.tools.VideoManager(output_dir=result_dir,
                                      framerate=24,
                                      automatic_build=True)


def main():
    init()
    SignedDistanceField = load_mesh_fast('./model/dragon.obj',
                                         n_grid,
                                         scale_ratio=1.0)
    print(SignedDistanceField.shape)

    init_sdf(SignedDistanceField)

    # F_x.from_numpy(ply3.get_array())

    # gui = ti.GUI("MPM3D", background_color=0x112F41)
    window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    i = 0

    # transform(0.2, 0.0, 0.2)

    while window.running:
        i = i + 1
        angle = float(i) * PI / 180.0
        # camera.position(5.0 * ti.cos(angle), 0.0, 5.0 * ti.sin(angle))
        camera.position(0.2 + ti.cos(angle), 0.8, 2.0 + ti.sin(angle))
        camera.lookat(0.2, 0.0, 0.2)

        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        for _ in range(steps):
            # pass
            substep()
            # update_co_position()

        # pos = F_x.to_numpy()
        # ball_center = T(np.array([co_position.to_numpy()]))

        scene.particles(F_x, radius=0.001, color=ORIANGE)

        # scene.particles(grid_pos, radius=0.005)

        scene.mesh(vertices, indices)

        canvas.scene(scene)
        video_manager.write_frame(window.get_image_buffer_as_numpy())
        window.show()

        # if export_file:
        #     writer = ti.tools.PLYWriter(num_vertices=n_particles)
        #     writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        #     writer.export_frame_ascii(gui.frame, export_file)

        # also can be replace by ti.ui.Scene()
        # gui.circle(ball_center[0], radius=45, color=0x068587)
        # gui.circles(T(pos), radius=1.5, color=0xED553B)
        # video_manager.write_frame(gui.get_image())
        # gui.show()


if __name__ == "__main__":
    main()
