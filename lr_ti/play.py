import taichi as ti

ti.init(arch=ti.cpu)


@ti.func
def num():
    return 1, 2, 3


@ti.kernel
def add() -> int:
    a, b, c = num()
    return a + b + c


vec3 = ti.types.vector(3, dtype=float)
sphere = ti.types.struct(center=vec3, r=float)


@ti.kernel
def test(x: sphere):
    x.r = 2


f = ti.field(dtype=ti.i32, shape=(3, 3, 3))


# equal to f = ti.field(dtype=ti.i32, shape=(9, ))
@ti.kernel
def fill():
    for i, j, k in f:
        f[i, j, k] = i


width, height = 640, 480
img = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))


@ti.kernel
def fill_img():
    for i, j in img:
        for k in ti.static(range(3)):
            img[i, j][k] = ti.random()


gui = ti.GUI(name="gray image", res=(width, height))
t = 0.0
while gui.running:
    fill_img()
    gui.set_image(img)
    gui.show()
