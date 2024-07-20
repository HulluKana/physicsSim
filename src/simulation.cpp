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

struct TetrahedronIndices {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};
std::vector<TetrahedronIndices> tetralize(const std::vector<dvec3> &origVerts, const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    // Algorithm for figuring out a half-decent regular tetrahedron that contains all vertices is from
    // https://computergraphics.stackexchange.com/questions/10533/how-to-compute-a-bounding-tetrahedron
    dvec3 maxPos = dvec3(std::numeric_limits<double>::min());
    dvec3 minPos = dvec3(std::numeric_limits<double>::max());
    for (const dvec3 &vert : origVerts) {
        maxPos = glm::max(maxPos, vert);
        minPos = glm::min(minPos, vert);
    }
    const dvec3 center = (maxPos + minPos) / 2.0;
    maxPos += 2.0 * (maxPos - center);
    minPos += 2.0 * (minPos - center);

    std::vector<dvec3> verts = origVerts;
    std::vector<TetrahedronIndices> tets(1);
    tets[0].a = verts.size();
    tets[0].b = verts.size() + 1;
    tets[0].c = verts.size() + 2;
    tets[0].d = verts.size() + 3;
    verts.push_back(maxPos);
    verts.push_back(dvec3(minPos.x, maxPos.y, minPos.z));
    verts.push_back(dvec3(minPos.x, minPos.y, maxPos.z));
    verts.push_back(dvec3(maxPos.x, minPos.y, minPos.z));

    std::vector<size_t> tetIdxsToRemove;
    std::unordered_set<uint32_t> removedTetCorners;
    for (size_t j = 0; j < origVerts.size(); j++) {
        tetIdxsToRemove.clear();
        removedTetCorners.clear();
        for (size_t i = 0; i < tets.size(); i++) {
            // Math for calculating the circumcenter is from this thread
            // https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
            constexpr double epsilon = std::numeric_limits<double>::epsilon();
            const dvec3 a = verts[tets[i].a];
            const dvec3 b = verts[tets[i].b];
            const dvec3 c = verts[tets[i].c];
            const dvec3 d = verts[tets[i].d];
            const double det = glm::dot(2.0 * (b - a), glm::cross(c - a, d - a)); 
            const dvec3 circumCenter = (det > -epsilon && det < epsilon) ? verts[tets[0].a] : a + (glm::dot(b - a, b - a) * glm::cross(c - a, d - a) + glm::dot(c - a, c - a) * glm::cross(d - a, b - a) + glm::dot(d - a, d - a) * glm::cross(b - a, c - a)) / det;
            if (glm::distance(verts[j], circumCenter) <= glm::distance(a, circumCenter)) {
                tetIdxsToRemove.push_back(i);
                removedTetCorners.insert(tets[i].a);
                removedTetCorners.insert(tets[i].b);
                removedTetCorners.insert(tets[i].c);
                removedTetCorners.insert(tets[i].d);
            }
        }
        for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);
        assert(removedTetCorners.size() == 0 || removedTetCorners.size() >= 3);
        if (removedTetCorners.size() > 0) {
            // Maybe try storing the faces of the deleted tets and use those to make new ones?
            TetrahedronIndices tet{.a = static_cast<uint32_t>(j)};
            uint32_t idx = 0;
            uint32_t firstOne;
            uint32_t secondOne;
            for (uint32_t cornerIdx : removedTetCorners) {
                if (idx == 0) firstOne = cornerIdx;
                if (idx == 1) secondOne = cornerIdx;
                tet.b = tet.c;
                tet.c = tet.d;
                tet.d = cornerIdx;
                if (idx >= 2) tets.push_back(tet);
                idx++;
            }
            tet.b = tet.c;
            tet.c = tet.d;
            tet.d = firstOne;
            tets.push_back(tet);
            tet.b = tet.c;
            tet.c = tet.d;
            tet.d = secondOne;
            tets.push_back(tet);
        }
    }
    std::cout << tets.size() << "\n";

    std::function<dvec3(const TetrahedronIndices &)> tetCentroid = [&](const TetrahedronIndices &tet) {
        const dvec3 a = verts[tet.a];
        const dvec3 b = verts[tet.b];
        const dvec3 c = verts[tet.c];
        const dvec3 d = verts[tet.d];
        return dvec3(a.x + b.x + c.x + d.x, a.y + b.y + c.y + d.y, a.z + b.z + c.z + d.z) / 4.0;
    };
    tetIdxsToRemove.clear();
    for (size_t i = 0; i < tets.size(); i++) {
        const dvec3 centroid = tetCentroid(tets[i]);
        constexpr std::array<dvec3, 6> rays = {dvec3(0.0, 1.0, 0.0), dvec3(0.0, -1.0, 0.0), dvec3(1.0, 0.0, 0.0), dvec3(-1.0, 0.0, 0.0), dvec3(0.0, 0.0, 1.0), dvec3(0.0, 0.0, -1.0)};
        size_t intersections = 0;
        for (const dvec3 &ray : rays) {
            double minDst = std::numeric_limits<double>::max();
            dvec3 closestTriangleNormal;
            bool foundIntersection = false;
            for (size_t j = mesh.firstIndex; j < mesh.firstIndex + mesh.indexCount; j += 3) {
                // Ray-triangle intersection code from
                // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
                constexpr double epsilon = std::numeric_limits<double>::epsilon();

                const dvec3 a = scene.vertices[scene.indices[j] + mesh.vertexOffset];
                const dvec3 b = scene.vertices[scene.indices[j + 1] + mesh.vertexOffset];
                const dvec3 c = scene.vertices[scene.indices[j + 2] + mesh.vertexOffset];

                const dvec3 edge1 = b - a;
                const dvec3 edge2 = c - a;
                const dvec3 rayCrossEdge2 = glm::cross(ray, edge2);
                const double det = glm::dot(edge1, rayCrossEdge2);
                if (det > -epsilon && det < epsilon) continue;

                const double invDet = 1.0 / det;
                const dvec3 s = centroid - a;
                const double u = invDet * glm::dot(s, rayCrossEdge2);
                if (u < 0.0 || u > 1.0) continue;

                const dvec3 sCrossEdge1 = glm::cross(s, edge1);
                const double v = invDet * glm::dot(ray, sCrossEdge1);
                if (v < 0.0 || u + v > 1.0) continue;

                const double dst = invDet * glm::dot(edge2, sCrossEdge1);
                if (dst > epsilon && dst < minDst) {
                    minDst = dst;
                    closestTriangleNormal = scene.normals[scene.indices[j] + mesh.vertexOffset];
                    foundIntersection = true;
                }
            }
            if (foundIntersection && glm::dot(ray, closestTriangleNormal) > 0.0) intersections++;
        }
        if (intersections < rays.size()) tetIdxsToRemove.push_back(i);
    }
    for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);

    tetIdxsToRemove.clear();
    for (size_t i = 0; i < tets.size(); i++) {
        if (tets[i].a >= origVerts.size() || tets[i].b >= origVerts.size() || tets[i].c >= origVerts.size() || tets[i].d >= origVerts.size()) tetIdxsToRemove.push_back(i);
    }
    for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);

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
        constraint.inverseStiffness = 100.0;
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
        constraint.inverseStiffness = 0.0001;
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
