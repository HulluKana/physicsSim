#include <cstdint>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/geometric.hpp>
#include <limits>
#include <tetrahedralization.hpp>

#include <bit>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

struct TriangleMesh {
    std::vector<dvec3> verts;
    std::vector<glm::uvec3> triangleIndices;
    std::vector<glm::uvec2> segmentIndices;
};
TriangleMesh getTriangeMeshFromGltfMesh(const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    TriangleMesh triangleMesh;

    constexpr double epsilon = 0.00001;
    std::vector<uint32_t> uniqueVertexIndices;
    std::unordered_map<uint32_t, uint32_t> meshVertToUniqueVertMap;
    for (uint32_t i = 0; i < mesh.vertexCount; i++) {
        bool unique = true;
        uint32_t uniqueVert = uniqueVertexIndices.size();
        for (size_t j = 0; j < uniqueVertexIndices.size(); j++) {
            if (glm::distance(scene.vertices[i + mesh.vertexOffset], scene.vertices[uniqueVertexIndices[j] + mesh.vertexOffset]) < epsilon) {
                unique = false;
                uniqueVert = j;
                break;
            }
        }
        if (unique) uniqueVertexIndices.push_back(i);
        meshVertToUniqueVertMap[i] = uniqueVert;
    }

    triangleMesh.verts.reserve(uniqueVertexIndices.size());
    for (size_t i = 0; i < uniqueVertexIndices.size(); i++) triangleMesh.verts.push_back(scene.vertices[uniqueVertexIndices[i] + mesh.vertexOffset]);

    triangleMesh.triangleIndices.reserve(mesh.indexCount / 3);
    for (uint32_t i = mesh.firstIndex; i < mesh.firstIndex + mesh.indexCount; i += 3) {
        triangleMesh.triangleIndices.push_back(glm::uvec3(meshVertToUniqueVertMap.at(scene.indices[i]),
                    meshVertToUniqueVertMap.at(scene.indices[i + 1]), meshVertToUniqueVertMap.at(scene.indices[i + 2])));
    }

    std::unordered_set<uint64_t> uniqueMeshSegments;
    for (const glm::uvec3 &triIndices : triangleMesh.triangleIndices) {
        std::array<glm::uvec2, 3> uniqueSegmentIndices = {glm::uvec2{triIndices.x, triIndices.y},
            {triIndices.x, triIndices.z}, {triIndices.y, triIndices.z}};
        for (glm::uvec2 &segmentIndices : uniqueSegmentIndices) {
            if (segmentIndices.x > segmentIndices.y) {
                const uint32_t temp = segmentIndices.y;
                segmentIndices.y = segmentIndices.x;
                segmentIndices.x = temp;
            }
            uniqueMeshSegments.insert(std::bit_cast<uint64_t>(segmentIndices));
        }
    }
    triangleMesh.segmentIndices.reserve(uniqueMeshSegments.size());
    for (uint64_t segment : uniqueMeshSegments) triangleMesh.segmentIndices.push_back(std::bit_cast<glm::uvec2>(segment));

    return triangleMesh;
}

dvec3 tetrahedronCircumcenter(const TetrahedronIndices &tet, const std::vector<dvec3> &verts)
{
    // Math for calculating the circumcenter is from this thread
    // https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    const dvec3 a = verts[tet.a];
    const dvec3 b = verts[tet.b];
    const dvec3 c = verts[tet.c];
    const dvec3 d = verts[tet.d];
    const double det = glm::dot(2.0 * (b - a), glm::cross(c - a, d - a)); 
    return (det > -epsilon && det < epsilon) ? a : a + (glm::dot(b - a, b - a) * glm::cross(c - a, d - a) +
            glm::dot(c - a, c - a) * glm::cross(d - a, b - a) + glm::dot(d - a, d - a) * glm::cross(b - a, c - a)) / det;
}

bool isPointInsideTetrahedron(const dvec3 &point, const TetrahedronIndices &tet, const std::vector<dvec3> &verts)
{
    const dvec3 circumCenter = tetrahedronCircumcenter(tet, verts);
    return glm::distance(point, circumCenter) <  glm::distance(verts[tet.a], circumCenter); 
}

std::unordered_set<uint64_t> getUniqueTetrahedronSegments(const TetrahedronMesh &tetMesh)
{
    std::unordered_set<uint64_t> uniqueTetrahedronSegments;
    for (const TetrahedronIndices &tet : tetMesh.tets) {
        std::array<glm::uvec2, 6> uniqueSegmentIndices = {glm::uvec2{tet.a, tet.b},
            {tet.a, tet.c}, {tet.b, tet.c},
            {tet.a, tet.d}, {tet.b, tet.d}, {tet.c, tet.d}};
        for (glm::uvec2 &segmentIndices : uniqueSegmentIndices) {
            if (segmentIndices.x > segmentIndices.y) {
                const uint32_t temp = segmentIndices.y;
                segmentIndices.y = segmentIndices.x;
                segmentIndices.x = temp;
            }
            uniqueTetrahedronSegments.insert(std::bit_cast<uint64_t>(segmentIndices));
        }
    }
    return uniqueTetrahedronSegments;
}

uint32_t countMissingSegments(const TriangleMesh &triangleMesh, const TetrahedronMesh &tetMesh)
{
    const std::unordered_set<uint64_t> uniqueTetrahedronSegments = getUniqueTetrahedronSegments(tetMesh);
    uint32_t missingSegments = 0;
    for (glm::uvec2 segment : triangleMesh.segmentIndices)
        if (uniqueTetrahedronSegments.find(std::bit_cast<uint64_t>(segment)) == uniqueTetrahedronSegments.end()) missingSegments++;
    return missingSegments;
}

uint32_t countNonDelaunayTetrahedrons(const TetrahedronMesh &tetMesh)
{
    uint32_t nonDelaunayTets = 0;
    for (const TetrahedronIndices &tet : tetMesh.tets) {
        for (size_t j = 0; j < tetMesh.verts.size(); j++) {
            if (tet.a == j || tet.b == j || tet.c == j || tet.d == j) continue;
            if (isPointInsideTetrahedron(tetMesh.verts[j], tet, tetMesh.verts)) {
                nonDelaunayTets++;
                break;
            }
        }
    }
    return nonDelaunayTets;
}

uint32_t countFlatTetrahedrons(const TetrahedronMesh &tetMesh)
{
    uint32_t flatTets = 0;
    for (const TetrahedronIndices &tet : tetMesh.tets) {
        const dvec3 a = tetMesh.verts[tet.a];
        const dvec3 b = tetMesh.verts[tet.b];
        const dvec3 c = tetMesh.verts[tet.c];
        const dvec3 d = tetMesh.verts[tet.d];
        const double det = glm::dot(2.0 * (b - a), glm::cross(c - a, d - a)); 
        if (fabs(det) < 0.000001) flatTets++;
    }
    return flatTets;
}

void insertVertexToTetrahedralizedMesh(TetrahedronMesh &tetMesh, const uint32_t vertIdx)
{
    struct Face {
        uint32_t a;
        uint32_t b;
        uint32_t c;
    };
    static std::vector<size_t> tetIdxsToRemove;
    static std::vector<Face> removedTetFaces;
    static std::vector<Face> uniqueTetFaces;
    static std::unordered_set<uint32_t> indicesToSkip;

    std::vector<TetrahedronIndices> &tets = tetMesh.tets;
    std::vector<dvec3> &verts = tetMesh.verts;

    tetIdxsToRemove.clear();
    removedTetFaces.clear();
    uniqueTetFaces.clear();
    indicesToSkip.clear();
    for (size_t i = 0; i < tets.size(); i++) {
        if (isPointInsideTetrahedron(verts[vertIdx], tets[i], verts)) {
            tetIdxsToRemove.push_back(i);
            removedTetFaces.emplace_back(Face{tets[i].a, tets[i].b, tets[i].c});
            removedTetFaces.emplace_back(Face{tets[i].a, tets[i].b, tets[i].d});
            removedTetFaces.emplace_back(Face{tets[i].b, tets[i].c, tets[i].d});
            removedTetFaces.emplace_back(Face{tets[i].a, tets[i].c, tets[i].d});
        }
    }
    for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);
    for (size_t i = 0; i < removedTetFaces.size(); i++) {
        if (indicesToSkip.find(i) != indicesToSkip.end()) continue;
        bool foundMatch = false;
        const Face &face1 = removedTetFaces[i];
        assert(face1.a != face1.b && face1.a != face1.c && face1.b != face1.c);
        for (size_t k = 0; k < removedTetFaces.size(); k++) {
            if (i == k) continue;
            const Face &face2 = removedTetFaces[k];
            if ((face1.a == face2.a || face1.a == face2.b || face1.a == face2.c) &&
                    (face1.b == face2.a || face1.b == face2.b || face1.b == face2.c) &&
                    (face1.c == face2.a || face1.c == face2.b || face1.c == face2.c)) {
                foundMatch = true;
                indicesToSkip.insert(k);
                break;
            }
        }
        if (!foundMatch) uniqueTetFaces.push_back(face1);
    }
    assert(uniqueTetFaces.size() == 0 || uniqueTetFaces.size() >= 3);
    for (const Face &face : uniqueTetFaces) {
        const TetrahedronIndices tet = {static_cast<uint32_t>(vertIdx), face.a, face.b, face.c};
        const dvec3 circumCenter = tetrahedronCircumcenter(tet, verts);
        const double radius = glm::distance(verts[vertIdx], circumCenter);
        bool nonDenaulnay = false;
        for (size_t i = 0; i < vertIdx; i++) {
            if (i == face.a || i == face.b || i == face.c) continue;
            if (glm::distance(verts[i], circumCenter) < radius - 0.000001) {
                nonDenaulnay = true;
                break;
            }
        }
        if (!nonDenaulnay) tets.push_back(tet);
    }
}

TetrahedronMesh createConvexHullTetrahedralMesh(const std::vector<dvec3> &origVerts)
{
    TetrahedronMesh tetMesh;
    tetMesh.verts = origVerts;
    std::vector<TetrahedronIndices> &tets = tetMesh.tets;
    std::vector<dvec3> &verts = tetMesh.verts;

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

    tets.emplace_back(TetrahedronIndices{static_cast<uint32_t>(verts.size()), static_cast<uint32_t>(verts.size()) + 1,
            static_cast<uint32_t>(verts.size()) + 2, static_cast<uint32_t>(verts.size()) + 3});
    verts.push_back(maxPos);
    verts.push_back(dvec3(minPos.x, maxPos.y, minPos.z));
    verts.push_back(dvec3(minPos.x, minPos.y, maxPos.z));
    verts.push_back(dvec3(maxPos.x, minPos.y, minPos.z));

    for (size_t j = 0; j < origVerts.size(); j++) insertVertexToTetrahedralizedMesh(tetMesh, j);

    std::vector<uint32_t> tetIdxsToRemove;
    verts.erase(verts.begin() + origVerts.size(), verts.end());
    for (size_t i = 0; i < tets.size(); i++)
        if (tets[i].a >= origVerts.size() || tets[i].b >= origVerts.size() || tets[i].c >= origVerts.size() || tets[i].d >= origVerts.size())
            tetIdxsToRemove.push_back(i);
    for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tets.erase(tets.begin() + *it);

    return tetMesh;
}

void edgeProtectTetrahedronMesh(TetrahedronMesh &tetMesh, TriangleMesh &triMesh)
{
    const TriangleMesh origTriMesh = triMesh; 

    std::unordered_map<uint32_t, std::vector<uint32_t>> segsIntersectedAtVertMap;
    for (size_t i = 0; i < origTriMesh.segmentIndices.size(); i++) {
        for (size_t j = 0; j < origTriMesh.segmentIndices.size(); j++) {
            if (i == j) continue;

            struct SegmentIntersectInfo {
                bool segment1isReverseOrder;
                bool segment2isReverseOrder;
            };
            std::array<SegmentIntersectInfo, 4> segmentIntersectInfos = {SegmentIntersectInfo{false, false}, {true, false}, {false, true}, {true, true}};
            for (SegmentIntersectInfo segIntInf : segmentIntersectInfos) {
                const uint32_t segment1startIdx = segIntInf.segment1isReverseOrder ? origTriMesh.segmentIndices[i].y : origTriMesh.segmentIndices[i].x;
                const uint32_t segment2startIdx = segIntInf.segment2isReverseOrder ? origTriMesh.segmentIndices[j].y : origTriMesh.segmentIndices[j].x;
                if (segment1startIdx == segment2startIdx) {
                    const uint32_t segment1endIdx = segIntInf.segment1isReverseOrder ? origTriMesh.segmentIndices[i].x : origTriMesh.segmentIndices[i].y;
                    const uint32_t segment2endIdx = segIntInf.segment2isReverseOrder ? origTriMesh.segmentIndices[j].x : origTriMesh.segmentIndices[j].y;
                    const dvec3 seg1dir = origTriMesh.verts[segment1endIdx] - origTriMesh.verts[segment1startIdx];
                    const dvec3 seg2dir = origTriMesh.verts[segment2endIdx] - origTriMesh.verts[segment2startIdx];
                    if (glm::dot(seg1dir, seg2dir) > 0.0) segsIntersectedAtVertMap[segment1startIdx].push_back(i);
                    break;
                }
            }
        }
    }

    for (const std::pair<const uint32_t, std::vector<uint32_t>> &vertIntSegs : segsIntersectedAtVertMap) {
        double dstToNearestVert = std::numeric_limits<double>::max();
        for (size_t k = 0; k < origTriMesh.verts.size(); k++) {
            if (vertIntSegs.first == k) continue;
            dstToNearestVert = std::min(dstToNearestVert, glm::distance(origTriMesh.verts[k], origTriMesh.verts[vertIntSegs.first]));
        }
        double localFeatureSize = dstToNearestVert;
        for (glm::uvec2 seg : origTriMesh.segmentIndices) {
            if (vertIntSegs.first == seg.x || vertIntSegs.first == seg.y) continue;
            const dvec3 &a = origTriMesh.verts[vertIntSegs.first];
            const dvec3 &b = origTriMesh.verts[seg.x];
            const dvec3 &c = origTriMesh.verts[seg.y];
            const double dstToSeg = glm::length(glm::cross(a - b, a - c)) / glm::length(c - b);
            localFeatureSize = std::min(localFeatureSize, dstToSeg);
        }

        double shortestSegLength = std::numeric_limits<double>::max();
        for (uint32_t seg : vertIntSegs.second) shortestSegLength = std::min(shortestSegLength,
                glm::distance(origTriMesh.verts[origTriMesh.segmentIndices[seg].x], origTriMesh.verts[origTriMesh.segmentIndices[seg].y]));

        const double radius = std::min(localFeatureSize, shortestSegLength / 3.0);
        for (uint32_t seg : vertIntSegs.second) {
            assert(origTriMesh.segmentIndices[seg].x == vertIntSegs.first || origTriMesh.segmentIndices[seg].y == vertIntSegs.first);
            if (vertIntSegs.first == origTriMesh.segmentIndices[seg].x) {
                triMesh.segmentIndices.emplace_back(glm::uvec2{origTriMesh.segmentIndices[seg].x, triMesh.verts.size()});
                triMesh.segmentIndices[seg].x = triMesh.verts.size();
                const dvec3 segDir = glm::normalize(origTriMesh.verts[origTriMesh.segmentIndices[seg].y] - origTriMesh.verts[vertIntSegs.first]);
                triMesh.verts.push_back(origTriMesh.verts[vertIntSegs.first] + segDir * radius);
            } else {
                triMesh.segmentIndices.emplace_back(glm::uvec2{origTriMesh.segmentIndices[seg].y, triMesh.verts.size()});
                triMesh.segmentIndices[seg].y = triMesh.verts.size();
                const dvec3 segDir = glm::normalize(origTriMesh.verts[origTriMesh.segmentIndices[seg].y] - origTriMesh.verts[vertIntSegs.first]);
                triMesh.verts.push_back(origTriMesh.verts[vertIntSegs.first] + segDir * radius);
            }

            tetMesh.verts.push_back(triMesh.verts[triMesh.verts.size() - 1]);
            insertVertexToTetrahedralizedMesh(tetMesh, tetMesh.verts.size() - 1);
        }
    }

    std::unordered_set<uint32_t> segmentsToSplit;
    for (uint32_t i = 0; i < triMesh.segmentIndices.size(); i++) segmentsToSplit.insert(i);
    while (segmentsToSplit.size() > 0) {
        const std::unordered_set<uint64_t> uniqueTetrahedronSegments = getUniqueTetrahedronSegments(tetMesh);
        for (auto it = segmentsToSplit.begin(); it != segmentsToSplit.end();) {
            const uint32_t idx = *it;
            if (uniqueTetrahedronSegments.find(std::bit_cast<uint64_t>(triMesh.segmentIndices[idx])) == uniqueTetrahedronSegments.end()) {
                it = segmentsToSplit.erase(it);
                continue;
            }

            triMesh.verts.push_back(triMesh.verts[triMesh.segmentIndices[idx].x] +
                    (triMesh.verts[triMesh.segmentIndices[idx].y] - triMesh.verts[triMesh.segmentIndices[idx].x]) / 2.0);
            tetMesh.verts.push_back(triMesh.verts[triMesh.verts.size() - 1]);
            insertVertexToTetrahedralizedMesh(tetMesh, tetMesh.verts.size() - 1);
            triMesh.segmentIndices.emplace_back(glm::uvec2{triMesh.segmentIndices[idx].x, triMesh.verts.size() - 1});
            triMesh.segmentIndices[idx].x = triMesh.verts.size() - 1;
            segmentsToSplit.insert(triMesh.segmentIndices.size() - 1);

            it++;
        }
    }
}

TetrahedronMesh tetralizeMesh(const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    TriangleMesh triangleMesh = getTriangeMeshFromGltfMesh(scene, mesh);
    TetrahedronMesh convexTetMesh = createConvexHullTetrahedralMesh(triangleMesh.verts);
    edgeProtectTetrahedronMesh(convexTetMesh, triangleMesh);

    std::cout << countMissingSegments(triangleMesh, convexTetMesh) << " segments out of " << triangleMesh.segmentIndices.size() << " are missing from tetrahedralization\n"; 
    std::cout << countNonDelaunayTetrahedrons(convexTetMesh) << " tetrahedrons out of " << convexTetMesh.tets.size() << " break the delaunay condition\n";
    std::cout << countFlatTetrahedrons(convexTetMesh) << " tetrahedrons out of " << convexTetMesh.tets.size() << " are flat\n";

    return convexTetMesh;
}
