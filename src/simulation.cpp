#include <cstdint>
#include <limits>
#include <simulation.hpp>
#include <iostream>

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

struct Tetrahedron {
    dvec3 x;
    dvec3 y;
    dvec3 z;
    dvec3 w;
};
std::vector<Tetrahedron> tetralize(const std::vector<dvec3> &verts)
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

    const dvec3 x = maxPos;
    const dvec3 y = dvec3(minPos.x, maxPos.y, minPos.z);
    const dvec3 z = dvec3(minPos.x, minPos.y, maxPos.z);
    const dvec3 w = dvec3(maxPos.x, minPos.y, minPos.z);
    return {{x, y, z, w}};
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
    for (size_t i = 0; i < uniqueVerts.size(); i++) uniqueVerts[i] = scene.vertices[uniqueVertexIndices[i] + obj.mesh.vertexOffset];
    std::vector<Tetrahedron> tetrahedrons = tetralize(uniqueVerts);

    for (const Tetrahedron &tet : tetrahedrons) {
        const uint32_t startingIdx = obj.pointMasses.size();
        obj.pointMasses.emplace_back(Pointmass{.pos = tet.x, .vel = dvec3(0.0)});
        obj.pointMasses.emplace_back(Pointmass{.pos = tet.y, .vel = dvec3(0.0)});
        obj.pointMasses.emplace_back(Pointmass{.pos = tet.z, .vel = dvec3(0.0)});
        obj.pointMasses.emplace_back(Pointmass{.pos = tet.w, .vel = dvec3(0.0)});

        for (uint32_t i = 0; i < 4; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                if (i == j) continue;
                Constraint constraint;
                constraint.pm1idx = startingIdx + i;
                constraint.pm2idx = startingIdx + j;
                constraint.length = glm::distance(obj.pointMasses[constraint.pm1idx].pos, obj.pointMasses[constraint.pm1idx].pos);
                constraint.stiffness = 100.0;
                obj.constraints.push_back(constraint);
            }
        }
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
