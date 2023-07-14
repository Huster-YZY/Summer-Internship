import taichi as ti

ti.init(arch=ti.cpu)
n = 10
x = ti.field(ti.f32, n)
x.fill(1)
f = [1, 2, 3, 4]


@ti.kernel
def test():
    # ti.loop_config(serialize=False)
    for i in range(4):
        print(f[i])


def go():
    for i in range(4):
        print(f[i])


go()
