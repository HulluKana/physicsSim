#include "vul_gltf_loader.hpp"
#include "vul_scene.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <glm/detail/qualifier.hpp>
#include <glm/gtx/quaternion.hpp>
#include <limits>
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

// Hash combine from https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hashCombine(std::size_t& seed) { }
template <typename T, typename... Rest>
inline void hashCombine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hashCombine(seed, rest...);
}

struct TetrahedronIndices {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};
std::vector<TetrahedronIndices> tetralize(const std::vector<dvec3> &verts, const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    // A lot of my ideas for this algorithm come from https://people.eecs.berkeley.edu/~jrs/papers/cdtbasic.pdf
    struct Face {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        bool reverseNormal;
    };
    std::unordered_map<uint64_t, Face> unfinishedFaces;
    for (uint32_t i = mesh.firstIndex; i < mesh.firstIndex + mesh.indexCount; i += 3) {
        std::array<uint32_t, 3> triIndices;
        std::array<dvec3, 3> triCorners = {scene.vertices[scene.indices[i] + mesh.vertexOffset], scene.vertices[scene.indices[i + 1] + mesh.vertexOffset], scene.vertices[scene.indices[i + 2] + mesh.vertexOffset]};
        for (uint32_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < verts.size(); k++) {
                constexpr double epsolon = 0.00001;
                if (glm::distance(triCorners[j], verts[k]) < epsolon) {
                    triIndices[j] = k;
                    break;
                }
            }
        }
        const Face face1 = {triIndices[0], triIndices[1], triIndices[2], false};
        const Face face2 = {triIndices[0], triIndices[1], triIndices[2], true};
        uint64_t seed = 0;
        hashCombine(seed, face1.a, face1.b, face1.c, face1.reverseNormal);
        unfinishedFaces[seed] = face1;
        seed = 0;
        hashCombine(seed, face2.a, face2.b, face2.c, face2.reverseNormal);
        unfinishedFaces[seed] = face2;
    }

    // This cursed math is obtained from running "Solve(Length((u1,u2,u3)-((k1, k2, k3)+(n1,n2,n3)*t))=Length((v1,v2,v3)-((k1,k2,k3)+(n1,n2,n3)*t)),t)" on geogebras cas calculator. I actually came up with this myself.
    // Unfortunately it doesnt seem to work too well and requires further testing
    /*
    std::function<double(const dvec3 &, const dvec3 &, const dvec3 &, const dvec3 &)> getCircumsphereRadius = [](const dvec3 &u, const dvec3 &v, const dvec3 &n, const dvec3 &k) {
        return (-2.0 * k.x * u.x + 2.0 * k.x * v.x - 2.0 * k.y * u.y + 2.0 * k.y * v.y - 2.0 * k.z * u.z + 2.0 * k.z * v.z + u.x * u.x + u.y * u.y + u.z * u.z - v.x * v.x - v.y * v.y - v.z * v.z) / (2.0 * n.x * u.x - 2.0 * n.x * v.x + 2.0 * n.y * u.y - 2.0 * n.y * v.y + 2.0 * n.z * u.z - 2.0 * n.z * v.z);
    };
    */
    std::vector<TetrahedronIndices> tets;
    for (auto it = unfinishedFaces.begin(); it != unfinishedFaces.end();) {
        const Face &face = it->second;
        const dvec3 &a = verts[face.a];
        const dvec3 &b = verts[face.b];
        const dvec3 &c = verts[face.c];
        const dvec3 middle = (a + b + c) / 3.0;
        dvec3 normal = glm::normalize(glm::cross(verts[face.b] - verts[face.a], verts[face.c] - verts[face.a]));
        if (face.reverseNormal) normal *= -1.0;
        double minDst = std::numeric_limits<double>::max();
        size_t minIdx;
        bool foundVert = false;
        for (size_t j = 0; j < verts.size(); j++) {
            constexpr double epsilon = 0.00001;
            const dvec3 &d = verts[j];
            if (glm::dot(d - middle, normal) <= 0.0) continue;
            if (glm::distance(a, d) < epsilon || glm::distance(b, d) < epsilon || glm::distance(c, d) < epsilon) continue;
            //const double dst = getCircumsphereRadius(a, verts[j], normal, middle);
            const double det = glm::dot(2.0 * (b - a), glm::cross(c - a, d - a)); 
            const dvec3 circumCenter = (det > -epsilon && det < epsilon) ? a : a + (glm::dot(b - a, b - a) * glm::cross(c - a, d - a) + glm::dot(c - a, c - a) * glm::cross(d - a, b - a) + glm::dot(d - a, d - a) * glm::cross(b - a, c - a)) / det;
            const double dst = glm::distance(d, circumCenter);
            if (dst < minDst) {
                minDst = dst;
                minIdx = j;
                foundVert = true;
            }
        }
        if (foundVert) {
            assert(face.a != minIdx && face.b != minIdx && face.c != minIdx);
            std::array<Face, 6> faces = {Face{face.a, face.b, static_cast<uint32_t>(minIdx), false}, Face{face.b, face.c, static_cast<uint32_t>(minIdx), false}, Face{face.a, static_cast<uint32_t>(minIdx), face.c, false}};
            for (uint32_t i = 0; i < 3; i++) {
                faces[i + 3] = faces[i];
                faces[i + 3].reverseNormal = true;
            }
            for (const Face &tetFace : faces) {
                uint64_t seed = 0;
                const auto tetFaceIt = unfinishedFaces.find(seed);
                assert(tetFaceIt != it);
                if (tetFaceIt != unfinishedFaces.end()) unfinishedFaces.erase(tetFaceIt);
                else unfinishedFaces[seed] = tetFace;
            }
            tets.emplace_back(TetrahedronIndices{face.a, face.b, face.c, static_cast<uint32_t>(minIdx)});
        }
        it = unfinishedFaces.erase(it);
    }
    return tets;
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
            if (diff.x < epsilon && diff.y < epsilon && diff.z < epsilon && true) {
                unique = false;
                pointMassIdx = j;
                break;
            }
        }
        if (unique) uniqueVertexIndices.push_back(i);
        obj.meshVertexIdxToPointMassIdx[i] = pointMassIdx;
    }

    std::vector<dvec3> uniqueVerts(uniqueVertexIndices.size());
    obj.pointMasses.reserve(uniqueVerts.size());
    for (size_t i = 0; i < uniqueVerts.size(); i++) uniqueVerts[i] = scene.vertices[uniqueVertexIndices[i] + obj.mesh.vertexOffset];
    for (const dvec3 &vert : uniqueVerts) obj.pointMasses.emplace_back(Pointmass{.pos = vert, .vel = dvec3(0.0)});
    auto k = glfwGetTime();
    std::vector<TetrahedronIndices> tetrahedrons = tetralize(uniqueVerts, scene, obj.mesh);
    std::cout << glfwGetTime() - k << "\n";

    std::unordered_set<uint64_t> uniquePmIdxPairs;
    for (const TetrahedronIndices &tet : tetrahedrons) {
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
    
    for (const TetrahedronIndices &tet : tetrahedrons) {
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

    std::cout << obj.dstConstraints.size() << " " << obj.volConstraints.size() << "\n";

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
            pm.pos.y = std::max(pm.pos.y, 0.0);
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
