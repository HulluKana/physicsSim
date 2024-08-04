#include <tetrahedralization.hpp>
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
    obj.energies.dstConstraintEnergy = 0.0;
    for (const DstConstraint &constraint : obj.dstConstraints) {
        const double length = glm::distance(obj.pointMasses[constraint.pm1idx].pos, obj.pointMasses[constraint.pm2idx].pos);
        obj.energies.dstConstraintEnergy += 0.5 * (1.0 + constraint.inverseStiffness) * (length - constraint.length) * (length - constraint.length);
    }
    obj.energies.volConstraintEnergy = 0.0;
    for (VolConstraint &constraint : obj.volConstraints) {
        Pointmass &pm1 = obj.pointMasses[constraint.pm1idx];
        Pointmass &pm2 = obj.pointMasses[constraint.pm2idx];
        Pointmass &pm3 = obj.pointMasses[constraint.pm3idx];
        Pointmass &pm4 = obj.pointMasses[constraint.pm4idx];
        const double C = glm::abs((glm::dot(glm::cross(pm2.pos - pm1.pos, pm3.pos - pm1.pos), pm4.pos - pm1.pos) - constraint.volume) / 6.0);
        const dvec3 gradC1 = glm::cross(pm4.pos - pm2.pos, pm3.pos - pm2.pos);
        const dvec3 gradC2 = glm::cross(pm3.pos - pm1.pos, pm4.pos - pm1.pos);
        const dvec3 gradC3 = glm::cross(pm4.pos - pm1.pos, pm2.pos - pm1.pos);
        const dvec3 gradC4 = glm::cross(pm2.pos - pm1.pos, pm3.pos - pm1.pos);
        const double lambda = C / (glm::dot(gradC1, gradC1) + glm::dot(gradC2, gradC2) + glm::dot(gradC3, gradC3) + glm::dot(gradC4, gradC4) + constraint.inverseStiffness);
        obj.energies.volConstraintEnergy += lambda * glm::length(gradC1);
        obj.energies.volConstraintEnergy += lambda * glm::length(gradC2);
        obj.energies.volConstraintEnergy += lambda * glm::length(gradC3);
        obj.energies.volConstraintEnergy += lambda * glm::length(gradC4);
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
            if (glm::distance(scene.vertices[i + obj.mesh.vertexOffset], scene.vertices[uniqueVertexIndices[j] + obj.mesh.vertexOffset]) < epsilon) {
                unique = false;
                pointMassIdx = j;
                break;
            }
        }
        if (unique) uniqueVertexIndices.push_back(i);
        obj.meshVertexIdxToPointMassIdx[i] = pointMassIdx;
    }

    const TetralizationResults tetRes = tetralizeMesh(scene, obj.mesh);
    obj.pointMasses.reserve(uniqueVertexIndices.size());
    for (const dvec3 &vert : tetRes.tetMesh.verts) obj.pointMasses.emplace_back(Pointmass{.pos = vert, .vel = dvec3(0.0)});
    obj.facetSegments = tetRes.facetMesh.facetSegments;

    std::unordered_set<uint64_t> uniquePmIdxPairs;
    for (const TetrahedronIndices &tet : tetRes.tetMesh.tets) {
        const std::array<uint32_t, 4> indices = {tet.a, tet.b, tet.c, tet.d};
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                if (i == j) continue;
                assert(i != j);
                const uint32_t smaller = indices[i] < indices[j] ? indices[i] : indices[j];
                const uint32_t larger = indices[i] > indices[j] ? indices[i] : indices[j];
                const uint64_t encoded = (static_cast<uint64_t>(smaller) << 32) + larger;
                uniquePmIdxPairs.insert(encoded);
            }
        }
    }
    for (uint64_t encodedPmIdxPair : uniquePmIdxPairs) {
        DstConstraint constraint;
        constraint.pm1idx = encodedPmIdxPair >> 32;
        constraint.pm2idx = encodedPmIdxPair << 32 >> 32;
        constraint.length = glm::distance(obj.pointMasses[constraint.pm1idx].pos, obj.pointMasses[constraint.pm2idx].pos);
        constraint.inverseStiffness = 0.0;
        obj.dstConstraints.push_back(constraint);
    }
    
    for (const TetrahedronIndices &tet : tetRes.tetMesh.tets) {
        VolConstraint constraint;
        constraint.pm1idx = tet.a;
        constraint.pm2idx = tet.b;
        constraint.pm3idx = tet.c;
        constraint.pm4idx = tet.d;
        const dvec3 x1 = obj.pointMasses[tet.a].pos;
        constraint.volume = glm::dot(glm::cross(obj.pointMasses[tet.b].pos - x1, obj.pointMasses[tet.c].pos - x1), obj.pointMasses[tet.d].pos - x1);
        constraint.inverseStiffness = 0.0;
        obj.volConstraints.push_back(constraint);
    }
    calculateEnergies(obj);

    return obj;
}

void simulate(Obj &obj, double frameTime, double deltaT)
{
    for (double spentDuration = 0.0; spentDuration < frameTime; spentDuration += deltaT) {
        for (Pointmass &pm : obj.pointMasses) {
            pm.vel -= dvec3(0.0, 9.81, 0.0) * deltaT;
            pm.oldPos = pm.pos;
            pm.pos += pm.vel * deltaT;
            if (pm.pos.y < 0.0) {
                // Kinda hacky
                pm.oldPos.y = -(pm.oldPos.y - pm.pos.y);
                pm.pos.y = 0.0;
            }
        }
        for (DstConstraint &constraint : obj.dstConstraints) {
            Pointmass &pm1 = obj.pointMasses[constraint.pm1idx];
            Pointmass &pm2 = obj.pointMasses[constraint.pm2idx];
            const double length = glm::distance(pm1.pos, pm2.pos);
            const dvec3 dir = (pm2.pos - pm1.pos) / length;
            const dvec3 change = 1.0 / (2.0 + constraint.inverseStiffness / (deltaT * deltaT)) * (length - constraint.length) * dir;
            pm1.pos += change;
            pm2.pos -= change;
        }
        for (VolConstraint &constraint : obj.volConstraints) {
            Pointmass &pm1 = obj.pointMasses[constraint.pm1idx];
            Pointmass &pm2 = obj.pointMasses[constraint.pm2idx];
            Pointmass &pm3 = obj.pointMasses[constraint.pm3idx];
            Pointmass &pm4 = obj.pointMasses[constraint.pm4idx];
            const double C = (glm::dot(glm::cross(pm2.pos - pm1.pos, pm3.pos - pm1.pos), pm4.pos - pm1.pos) - constraint.volume) / 6.0;
            const dvec3 gradC1 = glm::cross(pm4.pos - pm2.pos, pm3.pos - pm2.pos);
            const dvec3 gradC2 = glm::cross(pm3.pos - pm1.pos, pm4.pos - pm1.pos);
            const dvec3 gradC3 = glm::cross(pm4.pos - pm1.pos, pm2.pos - pm1.pos);
            const dvec3 gradC4 = glm::cross(pm2.pos - pm1.pos, pm3.pos - pm1.pos);
            const double lambda = -C / (glm::dot(gradC1, gradC1) + glm::dot(gradC2, gradC2) + glm::dot(gradC3, gradC3) + glm::dot(gradC4, gradC4) + constraint.inverseStiffness / (deltaT * deltaT));
            pm1.pos += lambda * gradC1;
            pm2.pos += lambda * gradC2;
            pm3.pos += lambda * gradC3;
            pm4.pos += lambda * gradC4;
        }
        for (Pointmass &pm : obj.pointMasses) pm.vel = (pm.pos - pm.oldPos) / deltaT;
    }

    calculateEnergies(obj);
}
