EXPORT = False
if EXPORT:
    export_file = "./frames/mpm3d.ply"
else:
    export_file = ""

import numpy as np
from plyImporter import PlyImporter
import taichi as ti

ti.init(arch=ti.gpu)

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 64, 25, 1e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 25, 8e-5

# ply2 = PlyImporter("/Users/YZY/g201/MPM/frames/mpm3d_000000.ply")
ply3 = PlyImporter("/Users/YZY/g201/MPM/model/bunny.ply")

# n_particles = n_grid**dim // 2**(dim - 1)
n_particles = ply3.get_count()
print(n_particles)
# exit(0)
dx = 1 / n_grid
inv_dx = n_grid
p_rho = 1
# p_vol = (dx * 0.5)**2
p_vol = (dx * 0.5)**dim
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E, nu = 40, 0.2
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
# F_J = ti.field(float, n_particles)
F = ti.Matrix.field(dim, dim, float, n_particles)

F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim


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
def Boundary():
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I][1] -= dt * gravity

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
    for i in range(n_particles):
        # F_x[i] = ti.Vector([ti.random() for _ in range(dim)]) * 0.4 + 0.15
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


def main():
    init()
    F_x.from_numpy(ply3.get_array())
    gui = ti.GUI("MPM3D", background_color=0x112F41)
    while gui.running and not gui.get_event(gui.ESCAPE):
        for _ in range(steps):
            substep()
        pos = F_x.to_numpy()
        if export_file:
            writer = ti.tools.PLYWriter(num_vertices=n_particles)
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
            writer.export_frame_ascii(gui.frame, export_file)
        # also can be replace by ti.ui.Scene()
        gui.circles(T(pos), radius=1.5, color=0xED553B)
        gui.show()


if __name__ == "__main__":
    main()
