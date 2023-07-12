import taichi as ti

ti.init(ti.cpu)
iteration_number = 20

# Naive method used to solve linear system
# A = ti.Matrix([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0], [2.0, 1.0, -1.0]])
# b = ti.Vector([23.0, 1.0, 20.0])
# x = ti.Vector.field(3, dtype=float, shape=())

# @ti.kernel
# def direct_solver():
#     t = A.inverse() @ b
#     x[None] = t

#Jacobi Iteration for scalar
n = 3
A = ti.field(float, shape=(n, n))
b = ti.field(float, shape=n)
x = ti.field(float, shape=n)
new_x = ti.field(float, shape=n)


@ti.kernel
def init():
    A[0, 0] = 10
    A[0, 1] = 3
    A[0, 2] = 1

    A[1, 0] = 2
    A[1, 1] = -10
    A[1, 2] = 3

    A[2, 0] = 1
    A[2, 1] = 3
    A[2, 2] = 10

    b[0] = 14
    b[1] = -5
    b[2] = 14


@ti.kernel
def Jacobi_iteration():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if j != i:
                r -= A[i, j] * x[j]

        new_x[i] = r / A[i, i]

    for i in range(n):
        x[i] = new_x[i]


@ti.kernel
def Jacobi_res() -> ti.f32:
    res = 0.0
    for i in range(n):
        r = b[i]
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r
    return res


# init()
# for i in range(iteration_number):
#     Jacobi_iteration()
#     print(Jacobi_res())
#     print(x)

#Jacobi Iteration for Vector


@ti.kernel
def J_iter():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] @ x[j]

        new_x[i] = A[i, i].inverse() @ r

    for i in range(n):
        x[i] = new_x[i]
