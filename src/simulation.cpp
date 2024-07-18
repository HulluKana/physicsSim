#include "vul_gltf_loader.hpp"
#include "vul_scene.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <glm/detail/qualifier.hpp>
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
    obj.energies.constraintEnergy = 0.0;
    for (const Constraint &constraint : obj.constraints) {
        const double length = glm::distance(obj.pointMasses[constraint.pm1idx].pos, obj.pointMasses[constraint.pm2idx].pos);
        obj.energies.constraintEnergy += 0.5 * constraint.stiffness * length * length;
    }
}

struct TetrahedronIndices {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};
std::vector<TetrahedronIndices> tetralize(const std::vector<dvec3> &verts, const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    // Algorithm for figuring out a half-decent regular tetrahedron that contains all vertices is from
    // https://computergraphics.stackexchange.com/questions/10533/how-to-compute-a-bounding-tetrahedron
    dvec3 maxPos = dvec3(std::numeric_limits<double>::min());
    dvec3 minPos = dvec3(std::numeric_limits<double>::max());
    for (const dvec3 &vert : verts) {
        maxPos = glm::max(maxPos, vert);
        minPos = glm::min(minPos, vert);
    }
    const dvec3 center = (maxPos + minPos) / 2.0;
    maxPos += 2.0 * (maxPos - center);
    minPos += 2.0 * (minPos - center);

    struct Tetrahedron {
        dvec3 a;
        dvec3 b;
        dvec3 c;
        dvec3 d;
    };
    std::vector<Tetrahedron> tets(1);
    tets[0].a = maxPos;
    tets[0].b = dvec3(minPos.x, maxPos.y, minPos.z);
    tets[0].c = dvec3(minPos.x, minPos.y, maxPos.z);
    tets[0].d = dvec3(maxPos.x, minPos.y, minPos.z);

    std::vector<size_t> tetIdxsToRemove;
    std::vector<dvec3> removedTetCorners;
    for (const dvec3 &vert : verts) {
        tetIdxsToRemove.clear();
        removedTetCorners.clear();
        for (size_t i = 0; i < tets.size(); i++) {
            // Math for calculating the circumcenter is from this thread
            // https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
            constexpr double epsilon = std::numeric_limits<double>::epsilon();
            const double det = glm::dot(2.0 * (tets[i].b - tets[i].a), glm::cross(tets[i].c - tets[i].a, tets[i].d - tets[i].a)); 
            const dvec3 circumCenter = (det > -epsilon && det < epsilon) ? tets[0].a : tets[i].a + (glm::dot(tets[i].b - tets[i].a, tets[i].b - tets[i].a) * glm::cross(tets[i].c - tets[i].a, tets[i].d - tets[i].a) + glm::dot(tets[i].c - tets[i].a, tets[i].c - tets[i].a) * glm::cross(tets[i].d - tets[i].a, tets[i].b - tets[i].a) + glm::dot(tets[i].d - tets[i].a, tets[i].d - tets[i].a) * glm::cross(tets[i].b - tets[i].a, tets[i].c - tets[i].a)) / det;
            if (glm::distance(vert, circumCenter) <= glm::distance(tets[i].a, circumCenter)) {
                tetIdxsToRemove.push_back(i);
                removedTetCorners.push_back(tets[i].a);
                removedTetCorners.push_back(tets[i].b);
                removedTetCorners.push_back(tets[i].c);
                removedTetCorners.push_back(tets[i].d);
            }
        }
        for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);
        if (removedTetCorners.size() > 0) {
            Tetrahedron tet{.a = vert, .b = removedTetCorners[0], .c = removedTetCorners[1], .d = removedTetCorners[2]};
            tets.push_back(tet);
            for (size_t i = 3; i < removedTetCorners.size(); i++) {
                tet.b = tet.c;
                tet.c = tet.d;
                tet.d = removedTetCorners[i];
                tets.push_back(tet);
            }
        }
    }

    std::function<dvec3(const Tetrahedron &)> tetCentroid = [](const Tetrahedron &tet) {
        return dvec3(tet.a.x + tet.b.x + tet.c.x + tet.d.x, tet.a.y + tet.b.y + tet.c.y + tet.d.y, tet.a.z + tet.b.z + tet.c.z + tet.d.z) / 4.0;
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
        bool aIsValid = false;
        bool bIsValid = false;
        bool cIsValid = false;
        bool dIsValid = false;
        for (const dvec3 &vert : verts) {
            constexpr double epsilon = 0.00001;
            if (glm::distance(tets[i].a, vert) > -epsilon && glm::distance(tets[i].a, vert) < epsilon) aIsValid = true;
            if (glm::distance(tets[i].b, vert) > -epsilon && glm::distance(tets[i].b, vert) < epsilon) bIsValid = true;
            if (glm::distance(tets[i].c, vert) > -epsilon && glm::distance(tets[i].c, vert) < epsilon) cIsValid = true;
            if (glm::distance(tets[i].d, vert) > -epsilon && glm::distance(tets[i].d, vert) < epsilon) dIsValid = true;
        }
        if (!aIsValid || !bIsValid || !cIsValid || !dIsValid) tetIdxsToRemove.push_back(i);
    }
    for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);

    for (auto it = tets.rbegin(); it < tets.rend(); it++) {
        const std::array<dvec3, 4> comparisons1 = {it->a, it->b, it->c, it->d};
        const dvec3 *compAddr = &it->a;
        for (const Tetrahedron &tet : tets) {
            if (&tet.a == compAddr) continue;
            const std::array<dvec3, 4> comparisons2 = {tet.a, tet.b, tet.c, tet.d};
            constexpr double epsilon = 0.00001;
            uint32_t matches = 0;
            for (const dvec3 &comp1 : comparisons1) {
                for (const dvec3 &comp2 : comparisons2) {
                    const double dst = glm::distance(comp1, comp2);
                    if (dst > -epsilon && dst < epsilon) {
                        matches++;
                        break;
                    }
                }
            }
            if (matches == comparisons1.size()) {
                tets.erase(std::next(it).base());
                break;
            }
        }
    }

    std::vector<TetrahedronIndices> tetIdxs;
    tetIdxs.reserve(tets.size());
    for (const Tetrahedron &tet : tets) {
        const std::array<dvec3, 4> corners = {tet.a, tet.b, tet.c, tet.d};
        std::array<uint32_t, 4> indices;
        uint32_t matchesFound = 0;
        for (size_t i = 0; i < verts.size(); i++) {
            constexpr double epsilon = 0.00001;
            for (size_t j = 0; j < corners.size(); j++) {
                const double dst = glm::distance(corners[j], verts[i]);
                if (dst > -epsilon && dst < epsilon) {
                    indices[j] = i;
                    matchesFound++;
                }
            }
        }
        assert(matchesFound == indices.size());
        tetIdxs.emplace_back(TetrahedronIndices{.a = indices[0], .b = indices[1], .c = indices[2], .d = indices[3]});
    }
    assert(tetIdxs.size() == tets.size());

    return tetIdxs;
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

    std::vector<dvec3> uniqueVerts(uniqueVertexIndices.size());
    obj.pointMasses.reserve(uniqueVerts.size());
    for (size_t i = 0; i < uniqueVerts.size(); i++) uniqueVerts[i] = scene.vertices[uniqueVertexIndices[i] + obj.mesh.vertexOffset];
    for (const dvec3 &vert : uniqueVerts) obj.pointMasses.emplace_back(Pointmass{.pos = vert, .vel = dvec3(0.0)});
    std::vector<TetrahedronIndices> tetrahedrons = tetralize(uniqueVerts, scene, obj.mesh);


    typedef std::pair<uint32_t, uint32_t> PmIdxPair;
    static_assert(sizeof(PmIdxPair) == sizeof(uint64_t));
    std::function<uint64_t(PmIdxPair)> PmIdxPairToUint64_t = [](PmIdxPair pmIdxPair) {return *reinterpret_cast<uint64_t *>(&pmIdxPair);};
    std::function<PmIdxPair(uint64_t)> uint64_tToPmIdxPair = [](uint64_t uint64) {return *reinterpret_cast<PmIdxPair *>(&uint64);};
    std::unordered_set<uint64_t> uniquePmIdxPairs;

    for (const TetrahedronIndices &tet : tetrahedrons) {
        const std::array<uint32_t, 4> indices = {tet.a, tet.b, tet.c, tet.d};
        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                if (i == j) continue;
                uniquePmIdxPairs.insert(PmIdxPairToUint64_t(PmIdxPair(indices[i], indices[j])));
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
    std::cout << obj.constraints.size() << "\n";

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
