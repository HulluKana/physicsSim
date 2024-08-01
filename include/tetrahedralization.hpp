#pragma once

#include <vul_scene.hpp>

typedef glm::vec<3, double> dvec3;

struct TetrahedronIndices {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};
struct TetrahedronMesh {
    std::vector<dvec3> verts;
    std::vector<TetrahedronIndices> tets;
};
TetrahedronMesh tetralizeMesh(const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh);
