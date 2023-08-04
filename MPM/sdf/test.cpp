#include <iostream>
#include <vector>
#include <cstring>
#include "./TriangleMeshDistance.h"
#include "./OBJ_Loader.h"
std::vector<std::array<double, 3>> vertices;
std::vector<std::array<int, 3>> triangles;
double x, y, z;
objl::Loader loader;

void initialize(std::string filename)
{
    auto flag = loader.LoadFile(filename);
    if (flag)
        std::cout << "Load Successfully!\n";
    else
    {
        std::cout << "Fail to Load!\n";
        return;
    }
    // Output for test
    // std::cout << loader.LoadedMeshes[0].Vertices.size() << std::endl;
    // std::cout << loader.LoadedVertices[0].Position << std::endl;
    // std::cout << loader.LoadedVertices[0].Normal << std::endl;

    int num_faces = loader.LoadedMeshes[0].Vertices.size() / 3;
    for (int i = 0; i < num_faces; i++)
    {
        std::array<int, 3> tri_idx;
        for (int j = 0; j < 3; j++)
        {
            int t = i * 3 + j;
            auto v = loader.LoadedMeshes[0].Vertices[t];
            auto idx = loader.LoadedMeshes[0].Indices[t];
            vertices.push_back({v.Position.X, v.Position.Y, v.Position.Z});
            tri_idx[j] = idx;
        }
        triangles.push_back(tri_idx);
    }

    return;
}

void test(std::string filename)
{
    // Initialize Loader
    objl::Loader Loader;

    // Load .obj File
    bool loadout = Loader.LoadFile(filename);

    // Check to see if it loaded

    // If so continue
    if (loadout)
    {
        // Create/Open e1Out.txt
        std::ofstream file("./e1Out.txt");

        // Go through each loaded mesh and out its contents
        for (int i = 0; i < Loader.LoadedMeshes.size(); i++)
        {
            // Copy one of the loaded meshes to be our current mesh
            objl::Mesh curMesh = Loader.LoadedMeshes[i];

            // Print Mesh Name
            file << "Mesh " << i << ": " << curMesh.MeshName << "\n";

            // Print Vertices
            file << "Vertices:\n";

            // Go through each vertex and print its number,
            //  position, normal, and texture coordinate
            for (int j = 0; j < curMesh.Vertices.size(); j++)
            {
                file << "V" << j << ": "
                     << "P(" << curMesh.Vertices[j].Position.X << ", " << curMesh.Vertices[j].Position.Y << ", " << curMesh.Vertices[j].Position.Z << ") "
                     << "N(" << curMesh.Vertices[j].Normal.X << ", " << curMesh.Vertices[j].Normal.Y << ", " << curMesh.Vertices[j].Normal.Z << ") "
                     << "TC(" << curMesh.Vertices[j].TextureCoordinate.X << ", " << curMesh.Vertices[j].TextureCoordinate.Y << ")\n";
            }

            // Print Indices
            file << "Indices:\n";

            // Go through every 3rd index and print the
            //	triangle that these indices represent
            for (int j = 0; j < curMesh.Indices.size(); j += 3)
            {
                file << "T" << j / 3 << ": " << curMesh.Indices[j] << ", " << curMesh.Indices[j + 1] << ", " << curMesh.Indices[j + 2] << "\n";
            }

            // Print Material
            file << "Material: " << curMesh.MeshMaterial.name << "\n";
            file << "Ambient Color: " << curMesh.MeshMaterial.Ka.X << ", " << curMesh.MeshMaterial.Ka.Y << ", " << curMesh.MeshMaterial.Ka.Z << "\n";
            file << "Diffuse Color: " << curMesh.MeshMaterial.Kd.X << ", " << curMesh.MeshMaterial.Kd.Y << ", " << curMesh.MeshMaterial.Kd.Z << "\n";
            file << "Specular Color: " << curMesh.MeshMaterial.Ks.X << ", " << curMesh.MeshMaterial.Ks.Y << ", " << curMesh.MeshMaterial.Ks.Z << "\n";
            file << "Specular Exponent: " << curMesh.MeshMaterial.Ns << "\n";
            file << "Optical Density: " << curMesh.MeshMaterial.Ni << "\n";
            file << "Dissolve: " << curMesh.MeshMaterial.d << "\n";
            file << "Illumination: " << curMesh.MeshMaterial.illum << "\n";
            file << "Ambient Texture Map: " << curMesh.MeshMaterial.map_Ka << "\n";
            file << "Diffuse Texture Map: " << curMesh.MeshMaterial.map_Kd << "\n";
            file << "Specular Texture Map: " << curMesh.MeshMaterial.map_Ks << "\n";
            file << "Alpha Texture Map: " << curMesh.MeshMaterial.map_d << "\n";
            file << "Bump Map: " << curMesh.MeshMaterial.map_bump << "\n";

            // Leave a space to separate from the next mesh
            file << "\n";
        }

        // Close File
        file.close();
    }
    // If not output an error
    else
    {
        // Create/Open e1Out.txt
        std::ofstream file("e1Out.txt");

        // Output Error
        file << "Failed to Load File. May have failed to find it or it was not an .obj file.\n";

        // Close File
        file.close();
    }

    // Exit the program
    return;
}
int main()
{
    initialize("../model/dragon.obj");
    // test("../model/cow.obj");

    // Initialize TriangleMeshDistance
    tmd::TriangleMeshDistance mesh_distance(vertices, triangles);

    // Query TriangleMeshDistance
    x = 1.0, y = 0.0, z = 0.0;
    tmd::Result result = mesh_distance.signed_distance({x, y, z});

    // Print result
    std::cout << "Signed distance: " << result.distance << std::endl;
    // // std::cout << "Nearest point: " << result.nearest_point << std::endl;
    // // std::cout << "Nearest entity: " << result.nearest_entity << std::endl;
    std::cout << "Nearest triangle index: " << result.triangle_id << std::endl;
}