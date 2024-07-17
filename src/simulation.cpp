#include <cstdint>
#include <functional>
#include <glm/geometric.hpp>
#include <simulation.hpp>
#include <iostream>
#include <unordered_set>

void calculateEnergies(Obj &obj)
{
    obj.energies.potentialEnergy = 0.0;
    obj.energies.kineticEnergy = 0.0;
    for (const Pointmass &pointmass : obj.pointMasses) {
        obj.energies.potentialEnergy += pointmass.pos.y * 9.81;
        obj.energies.kineticEnergy += 0.5 * glm::dot(pointmass.vel, pointmass.vel);
    }
    obj.energies.constraintEnergy = 0.0;
    for (const Constraint &constraint : obj.constraints) {
        const double length = glm::distance(obj.pointMasses[constraint.pm1idx].pos, obj.pointMasses[constraint.pm2idx].pos);
        obj.energies.constraintEnergy += 0.5 * constraint.stiffness * length * length;
    }
}

Obj getObjFromScene(const vul::Scene &scene, const std::string &objNodeName)
{
    Obj obj{};
    obj.mesh = scene.meshes[0];
    for (const vul::GltfLoader::GltfNode &node : scene.nodes) if (node.name == objNodeName) obj.mesh = scene.meshes[node.primMesh];

    constexpr double epsilon = 0.00001;
    std::vector<uint32_t> uniqueVertexIndices;
    for (uint32_t i = 0; i < obj.mesh.vertexCount; i++) {
        bool unique = true;
        uint32_t pointMassIdx = uniqueVertexIndices.size();
        for (size_t j = 0; j < uniqueVertexIndices.size(); j++) {
            const glm::vec3 diff = glm::abs(scene.vertices[i + obj.mesh.vertexOffset] - scene.vertices[uniqueVertexIndices[j] + obj.mesh.vertexOffset]);
            if (diff.x < epsilon && diff.y < epsilon && diff.z < epsilon) {
                unique = false;
                pointMassIdx = j;
                break;
            }
        }
        if (unique) uniqueVertexIndices.push_back(i);
        obj.meshVertexIdxToPointMassIdx[i] = pointMassIdx;
    }

    obj.pointMasses.resize(uniqueVertexIndices.size());
    for (size_t i = 0; i < obj.pointMasses.size(); i++) {
        obj.pointMasses[i].pos = scene.vertices[uniqueVertexIndices[i] + obj.mesh.vertexOffset];
        obj.pointMasses[i].vel = dvec3(0.0);
    }

    typedef std::pair<uint32_t, uint32_t> PmIdxPair;
    static_assert(sizeof(PmIdxPair) == sizeof(uint64_t));
    std::function<uint64_t(PmIdxPair)> PmIdxPairToUint64_t = [](PmIdxPair pmIdxPair) {return *reinterpret_cast<uint64_t *>(&pmIdxPair);};
    std::function<PmIdxPair(uint64_t)> uint64_tToPmIdxPair = [](uint64_t uint64) {return *reinterpret_cast<PmIdxPair *>(&uint64);};
    std::unordered_set<uint64_t> uniquePmIdxPairs;

    assert((obj.mesh.firstIndex + obj.mesh.indexCount) % 3 == 0);
    for (uint32_t i = obj.mesh.firstIndex; i < obj.mesh.firstIndex + obj.mesh.indexCount; i += 3) {
        const glm::vec<3, uint32_t> triVertIndices(obj.meshVertexIdxToPointMassIdx[scene.indices[i]], obj.meshVertexIdxToPointMassIdx[scene.indices[i + 1]],
                obj.meshVertexIdxToPointMassIdx[scene.indices[i + 2]]);
        const dvec3 &x = obj.pointMasses[triVertIndices.x].pos;
        const dvec3 &y = obj.pointMasses[triVertIndices.y].pos;
        const dvec3 &z = obj.pointMasses[triVertIndices.z].pos;
        double longestEdge = glm::max(glm::distance(x, y), glm::distance(x, z));
        longestEdge = glm::max(longestEdge, glm::distance(y, z));
        for (size_t j = 0; j < obj.pointMasses.size(); j++) {
            if (glm::distance(x, obj.pointMasses[j].pos) <= longestEdge + epsilon && j != triVertIndices.x) {
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(triVertIndices.x, j)));
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(j, triVertIndices.x)));
            }
            if (glm::distance(y, obj.pointMasses[j].pos) <= longestEdge + epsilon && j != triVertIndices.y) {
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(triVertIndices.y, j)));
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(j, triVertIndices.y)));
            }
            if (glm::distance(z, obj.pointMasses[j].pos) <= longestEdge + epsilon && j != triVertIndices.z) {
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(triVertIndices.z, j)));
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(j, triVertIndices.z)));
            }
        }
    }

    for (uint64_t encodedPmIdxPair : uniquePmIdxPairs) {
        PmIdxPair pmIdxPair = uint64_tToPmIdxPair(encodedPmIdxPair);
        Constraint constraint;
        constraint.pm1idx = pmIdxPair.first;
        constraint.pm2idx = pmIdxPair.second;
        constraint.length = glm::distance(obj.pointMasses[pmIdxPair.first].pos, obj.pointMasses[pmIdxPair.second].pos);
        constraint.stiffness = 100.0;
        obj.constraints.push_back(constraint);
    }

    calculateEnergies(obj);

    return obj;
}

void simulate(Obj &obj, double frameTime, double deltaT)
{
    for (double spentDuration = 0.0; spentDuration < frameTime; spentDuration += deltaT) {
        for (Pointmass &pm : obj.pointMasses) {
            pm.vel -= dvec3(0.0, 9.81, 0.0) * deltaT;
            pm.pos += pm.vel * deltaT;
            if (pm.pos.y <= 0.0) {
                pm.pos.y = 0.0;
                pm.vel.y *= -1.0;
            }
        }

        for (Constraint &constraint : obj.constraints) {
            Pointmass &pm1 = obj.pointMasses[constraint.pm1idx];
            const Pointmass &pm2 = obj.pointMasses[constraint.pm2idx];
            const double length = glm::distance(pm1.pos, pm2.pos);
            const dvec3 dir = (pm1.pos - pm2.pos) / length;
            pm1.vel -= dir * constraint.stiffness * (length - constraint.length) * deltaT;
        }
    }

    calculateEnergies(obj);
}
