#pragma once

#include "vul_gltf_loader.hpp"
#include "vul_scene.hpp"
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

typedef glm::vec<3, double> dvec3;
struct Pointmass{
    dvec3 pos;
    dvec3 oldPos;
    dvec3 vel;
};
struct DstConstraint {
    uint32_t pm1idx;
    uint32_t pm2idx;
    double length;
    double inverseStiffness;
};
struct VolConstraint {
    uint32_t pm1idx;
    uint32_t pm2idx;
    uint32_t pm3idx;
    uint32_t pm4idx;
    double volume;
    double inverseStiffness;
};
struct Energies {
    double kineticEnergy;
    double potentialEnergy;
    double dstConstraintEnergy;
    double volConstraintEnergy;
};
struct Obj {
    std::vector<Pointmass> pointMasses;
    std::vector<DstConstraint> dstConstraints;
    std::vector<VolConstraint> volConstraints;
    Energies energies;
    vul::GltfLoader::GltfPrimMesh mesh;
    std::unordered_map<uint32_t, uint32_t> meshVertexIdxToPointMassIdx;
    std::vector<std::vector<glm::uvec2>> facetSegments;
};

Obj getObjFromScene(const vul::Scene &scene, const std::string &objNodeName);
void simulate(Obj &obj, double frameTime, double deltaT);
