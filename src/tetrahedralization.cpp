#include <cassert>
#include <glm/common.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/geometric.hpp>
#include <limits>
#include <tetrahedralization.hpp>

#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <functional>

FacetMesh getFacetMeshFromGltfMesh(const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    FacetMesh facetMesh;

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
    facetMesh.verts.reserve(uniqueVertexIndices.size());
    for (size_t i = 0; i < uniqueVertexIndices.size(); i++) facetMesh.verts.push_back(scene.vertices[uniqueVertexIndices[i] + mesh.vertexOffset]);

    std::unordered_set<uint32_t> untestedTriangles;
    for (uint32_t i = mesh.firstIndex; i < mesh.firstIndex + mesh.indexCount; i += 3) untestedTriangles.insert(i);
    std::vector<std::vector<glm::uvec2>> encounteredSegments;
    std::vector<std::vector<glm::uvec3>> encounteredTriangles;
    for (uint32_t i = mesh.firstIndex; i < mesh.firstIndex + mesh.indexCount; i += 3) {
        if (untestedTriangles.find(i) == untestedTriangles.end()) continue;
        untestedTriangles.erase(i);
        encounteredSegments.emplace_back(std::vector<glm::uvec2>{});
        encounteredTriangles.emplace_back(std::vector<glm::uvec3>{});
        const std::array<uint32_t, 3> triIdxs = {meshVertToUniqueVertMap.at(scene.indices[i]), meshVertToUniqueVertMap.at(scene.indices[i + 1]),
                meshVertToUniqueVertMap.at(scene.indices[i + 2])};
        const dvec3 normal = glm::normalize(glm::cross(facetMesh.verts[triIdxs[1]] - facetMesh.verts[triIdxs[0]],
                    facetMesh.verts[triIdxs[2]] - facetMesh.verts[triIdxs[0]]));
        if (triIdxs[0] == triIdxs[1] || triIdxs[0] == triIdxs[2] || triIdxs[1] == triIdxs[2]) continue;

        std::function<void(const std::array<uint32_t, 3> &)> insertTriIdxs = [&encounteredSegments, &encounteredTriangles]
            (const std::array<uint32_t, 3> &triIdxs) {

            encounteredSegments[encounteredSegments.size() - 1].emplace_back(glm::uvec2{triIdxs[0], triIdxs[1]});
            encounteredSegments[encounteredSegments.size() - 1].emplace_back(glm::uvec2{triIdxs[0], triIdxs[2]});
            encounteredSegments[encounteredSegments.size() - 1].emplace_back(glm::uvec2{triIdxs[1], triIdxs[2]});
            encounteredTriangles[encounteredTriangles.size() - 1].emplace_back(glm::uvec3{triIdxs[0], triIdxs[1], triIdxs[2]});
            for (size_t k = encounteredSegments[encounteredSegments.size() - 1].size() - 3; k < encounteredSegments[encounteredSegments.size() - 1].size(); k++) {
                if (encounteredSegments[encounteredSegments.size() - 1][k].x > encounteredSegments[encounteredSegments.size() - 1][k].y) {
                    const uint32_t temp = encounteredSegments[encounteredSegments.size() - 1][k].x;
                    encounteredSegments[encounteredSegments.size() - 1][k].x = encounteredSegments[encounteredSegments.size() - 1][k].y;
                    encounteredSegments[encounteredSegments.size() - 1][k].y = temp;
                }
            }
        };
        insertTriIdxs(triIdxs);

        for (auto it = untestedTriangles.begin(); it != untestedTriangles.end();) {
            const std::array<uint32_t, 3> testTriIdxs = {meshVertToUniqueVertMap.at(scene.indices[*it]),
                meshVertToUniqueVertMap.at(scene.indices[*it + 1]), meshVertToUniqueVertMap.at(scene.indices[*it + 2])};
            const dvec3 testNormal = glm::normalize(glm::cross(facetMesh.verts[testTriIdxs[1]] - facetMesh.verts[testTriIdxs[0]],
                        facetMesh.verts[testTriIdxs[2]] - facetMesh.verts[testTriIdxs[0]]));
            if (testTriIdxs[0] == testTriIdxs[1] || testTriIdxs[0] == testTriIdxs[2] || testTriIdxs[1] == testTriIdxs[2]) {
                it = untestedTriangles.erase(it);
                continue;
            }
            constexpr double epsilon = 0.0003;
            if (glm::abs(glm::abs(glm::dot(normal, testNormal)) - 1.0) < epsilon) {
                insertTriIdxs(testTriIdxs);
                it = untestedTriangles.erase(it);
                continue;
            }
            it++;
        }
    }

    std::unordered_map<uint32_t, std::vector<uint32_t>> connectionGraph;
    std::unordered_set<uint32_t> toDoSet;
    std::unordered_set<uint32_t> doneSet;
    assert(encounteredSegments.size() == encounteredTriangles.size());
    for (const std::vector<glm::uvec2> &segments : encounteredSegments) {
        connectionGraph.clear();
        for (glm::uvec2 segment : segments) {
            connectionGraph[segment.x].push_back(segment.y);
            connectionGraph[segment.y].push_back(segment.x);
        }

        size_t subFacetStartIdx = facetMesh.facetSegments.size();
        facetMesh.facetSegments.emplace_back(std::vector<glm::uvec2>{segments[0]});
        for (size_t i = 1; i < segments.size(); i++) {
            doneSet.clear();
            toDoSet.clear();
            toDoSet.insert(segments[i].x);
            bool segmentIsConnected = false;
            while (!toDoSet.empty() && !segmentIsConnected) {
                const uint32_t vert = *toDoSet.begin();
                toDoSet.erase(toDoSet.begin());
                doneSet.insert(vert);
                for (uint32_t reachableVert : connectionGraph[vert]) {
                    for (size_t j = subFacetStartIdx; j < facetMesh.facetSegments.size(); j++) {
                        if (reachableVert == facetMesh.facetSegments[j][0].x) {
                            facetMesh.facetSegments[j].push_back(segments[i]);
                            segmentIsConnected = true;
                            break;
                        }
                    }
                    if (segmentIsConnected) break;
                    if (doneSet.find(reachableVert) == doneSet.end()) toDoSet.insert(reachableVert);
                }
            }
            if (!segmentIsConnected) facetMesh.facetSegments.emplace_back(std::vector<glm::uvec2>{segments[i]});
        }
    }

    for (const std::vector<glm::uvec3> &triangles : encounteredTriangles) {
        connectionGraph.clear();
        for (glm::uvec3 triangle : triangles) {
            connectionGraph[triangle.x].push_back(triangle.y);
            connectionGraph[triangle.x].push_back(triangle.z);
            connectionGraph[triangle.y].push_back(triangle.x);
            connectionGraph[triangle.y].push_back(triangle.z);
            connectionGraph[triangle.z].push_back(triangle.x);
            connectionGraph[triangle.z].push_back(triangle.y);
        }

        size_t subFacetStartIdx = facetMesh.facetTriangles.size();
        facetMesh.facetTriangles.emplace_back(std::vector<glm::uvec3>{triangles[0]});
        for (size_t i = 1; i < triangles.size(); i++) {
            doneSet.clear();
            toDoSet.clear();
            toDoSet.insert(triangles[i].x);
            bool triangleIsConnected = false;
            while (!toDoSet.empty() && !triangleIsConnected) {
                const uint32_t vert = *toDoSet.begin();
                toDoSet.erase(toDoSet.begin());
                doneSet.insert(vert);
                for (uint32_t reachableVert : connectionGraph[vert]) {
                    for (size_t j = subFacetStartIdx; j < facetMesh.facetTriangles.size(); j++) {
                        if (reachableVert == facetMesh.facetTriangles[j][0].x) {
                            facetMesh.facetTriangles[j].push_back(triangles[i]);
                            triangleIsConnected = true;
                            break;
                        }
                    }
                    if (triangleIsConnected) break;
                    if (doneSet.find(reachableVert) == doneSet.end()) toDoSet.insert(reachableVert);
                }
            }
            if (!triangleIsConnected) facetMesh.facetTriangles.emplace_back(std::vector<glm::uvec3>{triangles[i]});
        }
    }

    std::unordered_set<uint64_t> uniqueMeshSegments;
    const std::vector<std::vector<glm::uvec2>> nonUniqueSegments = facetMesh.facetSegments;
    facetMesh.facetSegments.clear();
    for (const std::vector<glm::uvec2> &segments : nonUniqueSegments) {
        facetMesh.facetSegments.emplace_back(std::vector<glm::uvec2>{});
        for (glm::uvec2 segment : segments) {
            if (std::count(segments.begin(), segments.end(), segment) == 1) {
                uniqueMeshSegments.insert(std::bit_cast<uint64_t>(glm::uvec2(segment.x, segment.y)));
                facetMesh.facetSegments[facetMesh.facetSegments.size() - 1].emplace_back(glm::uvec2{segment.x, segment.y});
            }
        }
    }
    facetMesh.facetSegments.shrink_to_fit();
    facetMesh.segmentIndices.reserve(uniqueMeshSegments.size());
    for (uint64_t segment : uniqueMeshSegments) facetMesh.segmentIndices.push_back(std::bit_cast<glm::uvec2>(segment));

    std::vector<std::unordered_set<uint32_t>> uniqueFacetIndices(facetMesh.facetSegments.size());
    for (size_t i = 0; i < facetMesh.facetSegments.size(); i++) {
        for (glm::uvec2 segment : facetMesh.facetSegments[i]) {
            uniqueFacetIndices[i].insert(segment.x);
            uniqueFacetIndices[i].insert(segment.y);
        }
    }

    facetMesh.facetIndices.resize(uniqueFacetIndices.size());
    for (size_t i = 0; i < uniqueFacetIndices.size(); i++) {
        for (uint32_t idx : uniqueFacetIndices[i]) facetMesh.facetIndices[i].push_back(idx);
    }

    return facetMesh;
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
    return glm::distance(point, circumCenter) < glm::distance(verts[tet.a], circumCenter); 
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

uint32_t countMissingSegments(const FacetMesh &facetMesh, const TetrahedronMesh &tetMesh)
{
    const std::unordered_set<uint64_t> uniqueTetrahedronSegments = getUniqueTetrahedronSegments(tetMesh);
    uint32_t missingSegments = 0;
    for (glm::uvec2 segment : facetMesh.segmentIndices)
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

void edgeProtectTetrahedronMesh(TetrahedronMesh &tetMesh, FacetMesh &facetMesh)
{
    const FacetMesh origFacetMesh = facetMesh; 

    std::unordered_map<uint32_t, std::vector<uint32_t>> segsIntersectedAtVertMap;
    for (size_t i = 0; i < origFacetMesh.segmentIndices.size(); i++) {
        for (size_t j = 0; j < origFacetMesh.segmentIndices.size(); j++) {
            if (i == j) continue;

            struct SegmentIntersectInfo {
                bool segment1isReverseOrder;
                bool segment2isReverseOrder;
            };
            std::array<SegmentIntersectInfo, 4> segmentIntersectInfos = {SegmentIntersectInfo{false, false}, {true, false}, {false, true}, {true, true}};
            for (SegmentIntersectInfo segIntInf : segmentIntersectInfos) {
                const uint32_t segment1startIdx = segIntInf.segment1isReverseOrder ? origFacetMesh.segmentIndices[i].y : origFacetMesh.segmentIndices[i].x;
                const uint32_t segment2startIdx = segIntInf.segment2isReverseOrder ? origFacetMesh.segmentIndices[j].y : origFacetMesh.segmentIndices[j].x;
                if (segment1startIdx == segment2startIdx) {
                    const uint32_t segment1endIdx = segIntInf.segment1isReverseOrder ? origFacetMesh.segmentIndices[i].x : origFacetMesh.segmentIndices[i].y;
                    const uint32_t segment2endIdx = segIntInf.segment2isReverseOrder ? origFacetMesh.segmentIndices[j].x : origFacetMesh.segmentIndices[j].y;
                    const dvec3 seg1dir = origFacetMesh.verts[segment1endIdx] - origFacetMesh.verts[segment1startIdx];
                    const dvec3 seg2dir = origFacetMesh.verts[segment2endIdx] - origFacetMesh.verts[segment2startIdx];
                    if (glm::dot(seg1dir, seg2dir) > 0.0) segsIntersectedAtVertMap[segment1startIdx].push_back(i);
                    break;
                }
            }
        }
    }

    std::function<void(uint32_t, uint32_t)> insertVertexInTheMiddleOfSegmentToAllRelevantFacets = [&](uint32_t segmentIdx, uint32_t vertexIdx) {
        for (size_t i = 0; i < facetMesh.facetIndices.size(); i++) {
            std::vector<uint32_t> &facetIndices = facetMesh.facetIndices[i];
            bool foundStart = false;
            bool foundEnd = false;
            for (uint32_t facetIdx : facetIndices) {
                if (facetMesh.segmentIndices[segmentIdx].x == facetIdx) foundStart = true;
                if (facetMesh.segmentIndices[segmentIdx].y == facetIdx) foundEnd = true;
                if (foundStart && foundEnd) break;
            }
            if (foundStart && foundEnd) {
                facetIndices.push_back(vertexIdx);
                for (size_t j = 0; j < facetMesh.facetSegments[i].size(); j++) if (facetMesh.facetSegments[i][j].x == facetMesh.segmentIndices[segmentIdx].x && facetMesh.facetSegments[i][j].y == facetMesh.segmentIndices[segmentIdx].y) facetMesh.facetSegments[i].erase(facetMesh.facetSegments[i].begin() + j);
                facetMesh.facetSegments[i].emplace_back(glm::uvec2{facetMesh.segmentIndices[segmentIdx].x, vertexIdx});
                facetMesh.facetSegments[i].emplace_back(glm::uvec2{facetMesh.segmentIndices[segmentIdx].y, vertexIdx});
            }
        }
    };

    for (const std::pair<const uint32_t, std::vector<uint32_t>> &vertIntSegs : segsIntersectedAtVertMap) {
        double dstToNearestVert = std::numeric_limits<double>::max();
        for (size_t k = 0; k < origFacetMesh.verts.size(); k++) {
            if (vertIntSegs.first == k) continue;
            dstToNearestVert = std::min(dstToNearestVert, glm::distance(origFacetMesh.verts[k], origFacetMesh.verts[vertIntSegs.first]));
        }
        double localFeatureSize = dstToNearestVert;
        for (glm::uvec2 seg : origFacetMesh.segmentIndices) {
            if (vertIntSegs.first == seg.x || vertIntSegs.first == seg.y) continue;
            const dvec3 &a = origFacetMesh.verts[vertIntSegs.first];
            const dvec3 &b = origFacetMesh.verts[seg.x];
            const dvec3 &c = origFacetMesh.verts[seg.y];
            const double dstToSeg = glm::length(glm::cross(a - b, a - c)) / glm::length(c - b);
            localFeatureSize = std::min(localFeatureSize, dstToSeg);
        }

        double shortestSegLength = std::numeric_limits<double>::max();
        for (uint32_t seg : vertIntSegs.second) shortestSegLength = std::min(shortestSegLength,
                glm::distance(origFacetMesh.verts[origFacetMesh.segmentIndices[seg].x], origFacetMesh.verts[origFacetMesh.segmentIndices[seg].y]));

        const double radius = std::min(localFeatureSize, shortestSegLength / 3.0);
        for (uint32_t seg : vertIntSegs.second) {
            assert(origFacetMesh.segmentIndices[seg].x == vertIntSegs.first || origFacetMesh.segmentIndices[seg].y == vertIntSegs.first);

            insertVertexInTheMiddleOfSegmentToAllRelevantFacets(seg, facetMesh.verts.size());

            if (vertIntSegs.first == origFacetMesh.segmentIndices[seg].x) {
                facetMesh.segmentIndices.emplace_back(glm::uvec2{origFacetMesh.segmentIndices[seg].x, facetMesh.verts.size()});
                facetMesh.segmentIndices[seg].x = facetMesh.segmentIndices[seg].y;
                facetMesh.segmentIndices[seg].y = facetMesh.verts.size();
                const dvec3 segDir = glm::normalize(origFacetMesh.verts[origFacetMesh.segmentIndices[seg].y] - origFacetMesh.verts[vertIntSegs.first]);
                facetMesh.verts.push_back(origFacetMesh.verts[vertIntSegs.first] + segDir * radius);
            } else {
                facetMesh.segmentIndices.emplace_back(glm::uvec2{origFacetMesh.segmentIndices[seg].y, facetMesh.verts.size()});
                facetMesh.segmentIndices[seg].y = facetMesh.verts.size();
                const dvec3 segDir = glm::normalize(origFacetMesh.verts[origFacetMesh.segmentIndices[seg].x] - origFacetMesh.verts[vertIntSegs.first]);
                facetMesh.verts.push_back(origFacetMesh.verts[vertIntSegs.first] + segDir * radius);
            }

            tetMesh.verts.push_back(facetMesh.verts[facetMesh.verts.size() - 1]);
            insertVertexToTetrahedralizedMesh(tetMesh, tetMesh.verts.size() - 1);
        }
    }

    std::unordered_set<uint32_t> segmentsToSplit;
    for (uint32_t i = 0; i < facetMesh.segmentIndices.size(); i++) segmentsToSplit.insert(i);
    while (segmentsToSplit.size() > 0) {
        for (auto it = segmentsToSplit.begin(); it != segmentsToSplit.end();) {
            constexpr double segmentMinLength = 0.02;
            const std::unordered_set<uint64_t> uniqueTetrahedronSegments = getUniqueTetrahedronSegments(tetMesh);
            const uint32_t idx = *it;
            if (uniqueTetrahedronSegments.find(std::bit_cast<uint64_t>(facetMesh.segmentIndices[idx])) != uniqueTetrahedronSegments.end()
                    || glm::distance(facetMesh.verts[facetMesh.segmentIndices[idx].x], facetMesh.verts[facetMesh.segmentIndices[idx].y]) < segmentMinLength) {
                it = segmentsToSplit.erase(it);
                continue;
            }

            insertVertexInTheMiddleOfSegmentToAllRelevantFacets(idx, facetMesh.verts.size());

            facetMesh.verts.push_back((facetMesh.verts[facetMesh.segmentIndices[idx].x] + facetMesh.verts[facetMesh.segmentIndices[idx].y]) / 2.0);
            tetMesh.verts.push_back(facetMesh.verts[facetMesh.verts.size() - 1]);
            insertVertexToTetrahedralizedMesh(tetMesh, tetMesh.verts.size() - 1);
            facetMesh.segmentIndices.emplace_back(glm::uvec2{facetMesh.segmentIndices[idx].y, facetMesh.verts.size() - 1});
            facetMesh.segmentIndices[idx].y = facetMesh.verts.size() - 1;
            segmentsToSplit.insert(facetMesh.segmentIndices.size() - 1);
        }
    }
}

struct IntersectionResults {
    bool intersects;
    double intersectionDistance;
};
IntersectionResults rayTriangleIntersection(const dvec3 &ray, const dvec3 &origin, const dvec3 &a, const dvec3 &b, const dvec3 &c)
{
    // Ray-triangle intersection code from
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    IntersectionResults results;
    results.intersects = false;
    results.intersectionDistance = std::numeric_limits<double>::max();
    constexpr double epsilon = std::numeric_limits<double>::epsilon();

    //const dvec3 a = scene.vertices[scene.indices[j] + mesh.vertexOffset];
    //const dvec3 b = scene.vertices[scene.indices[j + 1] + mesh.vertexOffset];
    //const dvec3 c = scene.vertices[scene.indices[j + 2] + mesh.vertexOffset];

    const dvec3 edge1 = b - a;
    const dvec3 edge2 = c - a;
    const dvec3 rayCrossEdge2 = glm::cross(ray, edge2);
    const double det = glm::dot(edge1, rayCrossEdge2);
    if (det > -epsilon && det < epsilon) return results;

    const double invDet = 1.0 / det;
    const dvec3 s = origin - a;
    const double u = invDet * glm::dot(s, rayCrossEdge2);
    if (u < 0.0 || u > 1.0) return results;

    const dvec3 sCrossEdge1 = glm::cross(s, edge1);
    const double v = invDet * glm::dot(ray, sCrossEdge1);
    if (v < 0.0 || u + v > 1.0) return results;

    const double dst = invDet * glm::dot(edge2, sCrossEdge1);
    if (dst > epsilon) {
        results.intersects = true;
        results.intersectionDistance = dst;
    }
    return results;
}

inline void hashCombine(std::size_t& seed) { }
template <typename T, typename... Rest>
inline void hashCombine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hashCombine(seed, rest...);
}

void recoverFacets(TetrahedronMesh &tetMesh, FacetMesh &facetMesh)
{
    struct Face {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        bool reverseNormal;
    };
    std::vector<TetrahedronIndices> removedTets;
    std::vector<glm::uvec3> removedTriangles;
    std::vector<glm::uvec3> uniqueRemovedTriangles;
    std::vector<std::vector<glm::uvec3>> cavities;
    std::vector<uint32_t> cavityIndices;
    std::unordered_map<uint64_t, Face> unfinishedFaces;
    std::unordered_set<uint64_t> finishedFaces;
    for (const std::vector<glm::uvec3> &triangles : facetMesh.facetTriangles) {
        assert(triangles.size() > 0);
        removedTets.clear();
        tetMesh.tets.erase(std::remove_if(tetMesh.tets.begin(), tetMesh.tets.end(), [&](const TetrahedronIndices &tet) {
            const std::array<glm::uvec2, 6> lines = {glm::uvec2{tet.a, tet.b}, {tet.a, tet.c},
                {tet.a, tet.d}, {tet.b, tet.c}, {tet.b, tet.d}, {tet.c, tet.d}};
            bool intersects = false;
            for (glm::uvec2 line : lines) {
                const dvec3 origin = tetMesh.verts[line.x];
                const dvec3 ray = glm::normalize(tetMesh.verts[line.y] - origin);
                for (const glm::uvec3 &tri : triangles) {
                    const dvec3 &a = facetMesh.verts[tri.x];
                    const dvec3 &b = facetMesh.verts[tri.y];
                    const dvec3 &c = facetMesh.verts[tri.z];
                    IntersectionResults intersectionResults = rayTriangleIntersection(ray, origin, a, b, c);
                    constexpr double epsilon = 0.00001;
                    if (intersectionResults.intersects && intersectionResults.intersectionDistance <
                            glm::length(tetMesh.verts[line.y] - origin) - epsilon) {
                        intersects = true;
                        removedTets.push_back(tet);
                        break;
                    }
                }
                if (intersects) break;
            }
            return intersects;
        }), tetMesh.tets.end());

        // The cavity splitter will only create one cavity per side and segments intersecting the facets are treated as if they don't. Fix those
        removedTriangles.clear(); 
        for (const TetrahedronIndices &tet : removedTets) {
            removedTriangles.emplace_back(glm::uvec3{tet.a, tet.b, tet.c});
            removedTriangles.emplace_back(glm::uvec3{tet.a, tet.b, tet.d});
            removedTriangles.emplace_back(glm::uvec3{tet.a, tet.c, tet.d});
            removedTriangles.emplace_back(glm::uvec3{tet.b, tet.c, tet.d});
        }
        uniqueRemovedTriangles.clear();
        for (const glm::uvec3 &tri : removedTriangles) {
            size_t triCount = std::count_if(removedTriangles.begin(), removedTriangles.end(), [&tri](const glm::uvec3 &uTri){
                return (tri.x == uTri.x || tri.x == uTri.y || tri.x == uTri.z) && (tri.y == uTri.x || tri.y == uTri.y
                        || tri.y == uTri.z) && (tri.z == uTri.x || tri.z == uTri.y || tri.z == uTri.z);
            });
            assert(triCount > 0);
            if (triCount == 1) uniqueRemovedTriangles.push_back(tri);
        }

        cavities.clear();
        cavities.emplace_back(std::vector<glm::uvec3>{});
        cavities.emplace_back(std::vector<glm::uvec3>{});
        for (uint32_t i = 0; i < 2; i++) {
            const dvec3 normal = glm::normalize(glm::cross(facetMesh.verts[triangles[0].z] - facetMesh.verts[triangles[0].x],
                    facetMesh.verts[triangles[0].y] - facetMesh.verts[triangles[0].x])) * (i == 0 ? 1.0 : -1.0);
            for (const glm::uvec3 &tri : uniqueRemovedTriangles) {
                /*
                //Uncomment this if you dont trust my shady shenanigans

                const std::array<glm::uvec2, 3> lines = {glm::uvec2{tri.x, tri.y}, {tri.x, tri.z}, {tri.y, tri.z}};
                bool intersects = false;
                for (glm::uvec2 line : lines) {
                    constexpr double epsilon = 0.00001;

                    const dvec3 origin = tetMesh.verts[line.x];
                    const dvec3 ray = glm::normalize(tetMesh.verts[line.y] - origin);
                    for (const glm::uvec3 &testTri : triangles) {
                        const dvec3 a = facetMesh.verts[testTri.x];
                        const dvec3 b = facetMesh.verts[testTri.y];
                        const dvec3 c = facetMesh.verts[testTri.z];
                        IntersectionResults intersectionResults = rayTriangleIntersection(ray, origin, a, b, c);
                        assert(!intersectionResults.intersects || intersectionResults.intersectionDistance >= glm::length(tetMesh.verts
                                    [line.y] - origin) - epsilon || intersectionResults.intersectionDistance < epsilon);
                    }
                }
                */
                constexpr double epsilon = 0.00001;
                const double side1 = glm::dot(tetMesh.verts[tri.x] - facetMesh.verts[triangles[0].x], normal);
                const double side2 = glm::dot(tetMesh.verts[tri.y] - facetMesh.verts[triangles[0].y], normal);
                const double side3 = glm::dot(tetMesh.verts[tri.z] - facetMesh.verts[triangles[0].z], normal);
                if (fabs(side1) > epsilon || fabs(side2) > epsilon || fabs(side3) > epsilon)
                    cavities[glm::sign(side1 + side2 + side3) > 0].push_back(tri);
            }
        }

        for (const std::vector<glm::uvec3> &cavity : cavities) {
            cavityIndices.clear();
            for (const glm::uvec3 &tri : cavity) {
                if (std::find(cavityIndices.begin(), cavityIndices.end(), tri.x) == cavityIndices.end()) cavityIndices.push_back(tri.x);
                if (std::find(cavityIndices.begin(), cavityIndices.end(), tri.y) == cavityIndices.end()) cavityIndices.push_back(tri.y);
                if (std::find(cavityIndices.begin(), cavityIndices.end(), tri.z) == cavityIndices.end()) cavityIndices.push_back(tri.z);
            }

            unfinishedFaces.clear();
            for (const glm::uvec3 tri : cavity) {
                Face face = {tri.x, tri.y, tri.z, false};
                if (face.a > face.b) std::swap(face.a, face.b);
                if (face.b > face.c) std::swap(face.b, face.c);
                if (face.a > face.b) std::swap(face.a, face.b);
                uint64_t seed = 0;
                hashCombine(seed, face.a, face.b, face.c, face.reverseNormal);
                unfinishedFaces[seed] = face;
                seed = 0;
                face.reverseNormal = true;
                hashCombine(seed, face.a, face.b, face.c, face.reverseNormal);
                unfinishedFaces[seed] = face;
            }

            finishedFaces.clear();
            while (!unfinishedFaces.empty()) {
                const Face face = unfinishedFaces.begin()->second;
                finishedFaces.insert(unfinishedFaces.begin()->first);
                unfinishedFaces.erase(unfinishedFaces.begin());

                const dvec3 &a = tetMesh.verts[face.a];
                const dvec3 &b = tetMesh.verts[face.b];
                const dvec3 &c = tetMesh.verts[face.c];
                const dvec3 middle = (a + b + c) / 3.0;
                dvec3 normal = glm::normalize(glm::cross(tetMesh.verts[face.b] - tetMesh.verts[face.a],
                            tetMesh.verts[face.c] - tetMesh.verts[face.a]));
                if (face.reverseNormal) normal *= -1.0;
                double minDst = std::numeric_limits<double>::max();
                size_t minIdx;
                bool foundVert = false;
                for (uint32_t j : cavityIndices) {
                    constexpr double epsilon = 0.00001;
                    const dvec3 &d = tetMesh.verts[j];
                    if (glm::dot(d - middle, normal) <= 0.0) continue;
                    if (glm::distance(a, d) < epsilon || glm::distance(b, d) < epsilon || glm::distance(c, d) < epsilon) continue;
                    /*
                    bool visible = true;
                    for (const glm::uvec3 &tri : cavity) {
                        IntersectionResults results = rayTriangleIntersection(glm::normalize(d - middle), middle, tetMesh.verts[tri.x], tetMesh.verts[tri.y], tetMesh.verts[tri.z]);
                        if (results.intersects) {
                            visible = false;
                            break;
                        }
                    }
                    if (!visible) continue;
                    */
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
                    const dvec3 centroid = (tetMesh.verts[face.a] + tetMesh.verts[face.b] + tetMesh.verts[face.c] + tetMesh.verts[minIdx]) / 4.0;
                    std::array<Face, 3> faces = {Face{face.a, face.b, static_cast<uint32_t>(minIdx), false}, Face{face.b, face.c, static_cast<uint32_t>(minIdx), false}, Face{face.a, static_cast<uint32_t>(minIdx), face.c, false}};
                    for (Face &tetFace : faces) {
                        if (tetFace.a > tetFace.b) std::swap(tetFace.a, tetFace.b);
                        if (tetFace.b > tetFace.c) std::swap(tetFace.b, tetFace.c);
                        if (tetFace.a > tetFace.b) std::swap(tetFace.a, tetFace.b);
                        const dvec3 center = (tetMesh.verts[tetFace.a] + tetMesh.verts[tetFace.b] + tetMesh.verts[tetFace.c]) / 3.0;
                        if (glm::dot(glm::normalize(centroid - center), normal) > 0.0) tetFace.reverseNormal = !face.reverseNormal;
                        uint64_t seed = 0;
                        hashCombine(seed, tetFace.a, tetFace.b, tetFace.c, tetFace.reverseNormal);
                        const bool isFinished1 = finishedFaces.contains(seed);
                        const auto tetFaceIt1 = unfinishedFaces.find(seed);
                        seed = 0;
                        hashCombine(seed, tetFace.a, tetFace.b, tetFace.c, !tetFace.reverseNormal);
                        const bool isFinished2 = finishedFaces.contains(seed);
                        const auto tetFaceIt2 = unfinishedFaces.find(seed);
                        finishedFaces.insert(seed);
                        if (tetFaceIt1 != unfinishedFaces.end() || tetFaceIt2 != unfinishedFaces.end() || isFinished1 || isFinished2) {
                            if (tetFaceIt1 != unfinishedFaces.end()) unfinishedFaces.erase(tetFaceIt1);
                            if (tetFaceIt2 != unfinishedFaces.end()) unfinishedFaces.erase(tetFaceIt2);
                        }
                        else unfinishedFaces[seed] = tetFace;
                    }
                    tetMesh.tets.emplace_back(TetrahedronIndices{face.a, face.b, face.c, static_cast<uint32_t>(minIdx)});
                }
            }
        }
    }
}

void deleteUnnecessaryTetrahedrons(TetrahedronMesh &tetMesh, const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    std::vector<uint32_t> tetIdxsToRemove;
    for (size_t i = 0; i < tetMesh.tets.size(); i++) {
        const TetrahedronIndices &tet = tetMesh.tets[i];
        const dvec3 centroid = (tetMesh.verts[tet.a] + tetMesh.verts[tet.b] + tetMesh.verts[tet.c] + tetMesh.verts[tet.d]) / 4.0;
        std::array<dvec3, 4> rays = {(tetMesh.verts[tet.a] + tetMesh.verts[tet.b] + tetMesh.verts[tet.c]) / 3.0,
            (tetMesh.verts[tet.a] + tetMesh.verts[tet.b] + tetMesh.verts[tet.d]) / 3.0, (tetMesh.verts[tet.b] + tetMesh.verts[tet.c]
            + tetMesh.verts[tet.d]) / 3.0, (tetMesh.verts[tet.a] + tetMesh.verts[tet.c] + tetMesh.verts[tet.d]) / 3.0};
        for (dvec3 &ray : rays) {
            ray = glm::normalize(ray - centroid);
            bool foundIntersection = false;
            double minDst = std::numeric_limits<double>::max();
            dvec3 closestTriangleNormal{};
            for (uint32_t j = mesh.firstIndex; j < mesh.firstIndex + mesh.indexCount; j += 3) {
                const dvec3 a = scene.vertices[scene.indices[j] + mesh.vertexOffset];
                const dvec3 b = scene.vertices[scene.indices[j + 1] + mesh.vertexOffset];
                const dvec3 c = scene.vertices[scene.indices[j + 2] + mesh.vertexOffset];
                IntersectionResults intersectionResults = rayTriangleIntersection(ray, centroid, a, b, c);
                if (intersectionResults.intersects && intersectionResults.intersectionDistance < minDst) {
                    foundIntersection = true;
                    minDst = intersectionResults.intersectionDistance;
                    closestTriangleNormal = glm::normalize(glm::cross(b - a, c - a));
                }
            }
            if (!foundIntersection || glm::dot(ray, closestTriangleNormal) <= 0.0) {
                tetIdxsToRemove.push_back(i);
                break;
            }
        }
    }
    for (auto it = tetIdxsToRemove.rbegin(); it < tetIdxsToRemove.rend(); it++) tetMesh.tets.erase(tetMesh.tets.begin() + *it);
}

TetralizationResults tetralizeMesh(const vul::Scene &scene, const vul::GltfLoader::GltfPrimMesh &mesh)
{
    FacetMesh facetMesh = getFacetMeshFromGltfMesh(scene, mesh);
    TetrahedronMesh tetMesh = createConvexHullTetrahedralMesh(facetMesh.verts);
    edgeProtectTetrahedronMesh(tetMesh, facetMesh);
    recoverFacets(tetMesh, facetMesh);
    deleteUnnecessaryTetrahedrons(tetMesh, scene, mesh);

    std::cout << countMissingSegments(facetMesh, tetMesh) << " segments out of " << facetMesh.segmentIndices.size() << " are missing from tetrahedralization\n"; 
    std::cout << countNonDelaunayTetrahedrons(tetMesh) << " tetrahedrons out of " << tetMesh.tets.size() << " break the delaunay condition\n";
    std::cout << countFlatTetrahedrons(tetMesh) << " tetrahedrons out of " << tetMesh.tets.size() << " are flat\n";

    return TetralizationResults{tetMesh, facetMesh};
}
