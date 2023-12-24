# Introduction

This repository is mainly about basic physical simulation, such as mass-spring system, FEM and MPM. This is also a record of my summer internship.

If you want to excute my code, please install the [Taichi](https://github.com/taichi-dev/taichi) at first.

<h4 align=center>Julia Fractal</h4>
<p align=center>
 <img src=./video/julia.gif/>
</p>

<h4 align=center>Cloth Simulation</h4>

![](./video/mass_spring.gif)




|Implicit Solver|Explicit Solver|
|--|--|
| <img src=./video/implicit_mass_spring.gif/> | <img src=./video/explicit_mass_spring.gif/> |

<h4 align=center>MPM (Material Point Method) </h4>

|fluid|jelly|snow|
|--|--|--|
| <img src=./video/fluid.gif/> | <img src=./video/jelly.gif/> |<img src=./video/snow.gif>|

<h4 align=center>3DMPM with Collision Treatment </h4>

<p align=center>
 <img src=./video/3dmpm_collision.gif/>
</p>

<h4 align=center>Load and visualize the 3D Model  </h4>

<p align=center>
 <img src=./video/model.gif/ height=300>
</p>


<h4 align=center>Signed Distance Field</h4>

The results are computed by using [Mesh2SDF](https://github.com/wang-ps/mesh2sdf).

|bunny|dragon|
|--|--|
| <img src=./video/bunny_sdf.gif/> | <img src=./video/dragon_sdf.gif/> |


<h4 align=center>Collision Objects</h4>


<p align=center>
 <img src=./video/collision_static.gif/>
</p>


|dynamic|marching cubes|fracture|
|--|--|--|
| <img src=./video/dynamic_co.gif/> | <img src=./video/marching_cube.gif/> |<img src=./video/fracture.gif>|
