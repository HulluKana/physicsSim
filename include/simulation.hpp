#pragma once

#include "vul_gltf_loader.hpp"
#include "vul_scene.hpp"
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

typedef glm::vec<3, double> dvec3;
struct Pointmass{
    dvec3 pos;
    dvec3 vel;
};
struct Constraint {
    uint32_t pm1idx;
    uint32_t pm2idx;
    double length;
    double stiffness;
};
struct Energies {
    double kineticEnergy;
    double potentialEnergy;
    double constraintEnergy;
};
struct Obj {
    std::vector<Pointmass> pointMasses;
    std::vector<Constraint> constraints;
    Energies energies;
    vul::GltfLoader::GltfPrimMesh mesh;
    std::unordered_map<uint32_t, uint32_t> meshVertexIdxToPointMassIdx;
};

Obj getObjFromScene(const vul::Scene &scene, const std::string &objNodeName);
void simulate(Obj &obj, double frameTime, double deltaT);
