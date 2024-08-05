#include "simulation.hpp"
#include "vul_comp_pipeline.hpp"
#include "vul_gltf_loader.hpp"
#include "vul_pipeline.hpp"
#include "vulkano_program.hpp"
#include <host_device.hpp>
#include <rendering.hpp>
#include <imgui.h>
#include <iostream>
#include <vulkan/vulkan_core.h>

RenderingResources initializeRenderingStuff(const std::string &modelDir, const std::string &modelFileName, vul::Vulkano &vulkano)
{
    RenderingResources renderingResources{};

    std::string modelPath = modelDir;
    if (modelPath[modelPath.length() - 1] != '/') modelPath += '/';
    modelPath += modelFileName;
    vulkano.loadScene(modelPath, modelDir, vul::Scene::WantedBuffers{.vertex = true, .index = true, .normal = true, .tangent = false, .uv = false, .material = true, .primInfo = false});
    vulkano.scene.vertexBuffer->reallocElsewhere(false);

    std::vector<std::shared_ptr<vul::VulDescriptorSet>> descSets(vul::VulSwapChain::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < descSets.size(); i++) {
        renderingResources.ubos[i] = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
        renderingResources.ubos[i]->keepEmpty(sizeof(Ubo), 1);
        renderingResources.ubos[i]->createBuffer(false, vul::VulBuffer::usage_ubo);

        std::vector<vul::Vulkano::Descriptor> descs;
        vul::Vulkano::Descriptor desc;
        desc.type = vul::Vulkano::DescriptorType::ubo;
        desc.stages = {vul::Vulkano::ShaderStage::vert, vul::Vulkano::ShaderStage::frag};
        desc.count = 1;
        desc.content = renderingResources.ubos[i].get();
        descs.push_back(desc);

        desc.type = vul::Vulkano::DescriptorType::ssbo;
        desc.stages = {vul::Vulkano::ShaderStage::frag};
        desc.content = vulkano.scene.materialBuffer.get();
        descs.push_back(desc);

        descSets[i] = vulkano.createDescriptorSet(descs);
    }

    vul::VulPipeline::PipelineConfigInfo defaultPipelineConfig{};
    defaultPipelineConfig.depthAttachmentFormat = vulkano.vulRenderer.getDepthFormat();
    defaultPipelineConfig.colorAttachmentFormats = {vulkano.vulRenderer.getSwapChainColorFormat()};
    defaultPipelineConfig.setLayouts = {descSets[0]->getLayout()->getDescriptorSetLayout()};
    defaultPipelineConfig.bindingDescriptions = {{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}, {1, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};
    defaultPipelineConfig.attributeDescriptions = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0}};

    std::vector<std::shared_ptr<DefaultPC>> defaultPushConstants(vulkano.scene.nodes.size());
    vul::Vulkano::RenderData defaultRenderData{};
    defaultRenderData.enable = false;
    defaultRenderData.pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "default.vert.spv", "default.frag.spv", defaultPipelineConfig);
    defaultRenderData.is3d = true;
    defaultRenderData.sampleFromDepth = false;
    defaultRenderData.depthImageMode = vul::VulRenderer::DepthImageMode::clearPreviousDiscardCurrent;
    defaultRenderData.swapChainImageMode = vul::VulRenderer::SwapChainImageMode::clearPreviousStoreCurrent;
    defaultRenderData.swapChainClearColor = glm::vec4(0.529f, 0.808f, 0.922f, 1.0f);
    defaultRenderData.depthClearColor = 1.0f;
    defaultRenderData.indexBuffer = vulkano.scene.indexBuffer.get();
    defaultRenderData.vertexBuffers = {vulkano.scene.vertexBuffer.get(), vulkano.scene.normalBuffer.get()};
    for (int i = 0; i < vul::VulSwapChain::MAX_FRAMES_IN_FLIGHT; i++) defaultRenderData.descriptorSets[i] = {descSets[i]};
    for (size_t i = 0; i < vulkano.scene.nodes.size(); i++) {
        const vul::GltfLoader::GltfNode &node = vulkano.scene.nodes[i];
        const vul::GltfLoader::GltfPrimMesh &mesh = vulkano.scene.meshes[node.primMesh];
        defaultPushConstants[i] = std::make_shared<DefaultPC>();
        defaultPushConstants[i]->matIdx = mesh.materialIndex;

        vul::VulPipeline::DrawData drawData{};
        drawData.pPushData = static_cast<std::shared_ptr<void>>(defaultPushConstants[i]);
        drawData.pushDataSize = sizeof(DefaultPC);
        drawData.vertexOffset = mesh.vertexOffset;
        drawData.firstIndex = mesh.firstIndex;
        drawData.indexCount = mesh.indexCount;
        defaultRenderData.drawDatas.push_back(drawData);
    }

    vul::VulPipeline::PipelineConfigInfo wireframePipelineConfig = defaultPipelineConfig;
    wireframePipelineConfig.bindingDescriptions = {{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};
    wireframePipelineConfig.attributeDescriptions = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    wireframePipelineConfig.polygonMode = VK_POLYGON_MODE_LINE;
    wireframePipelineConfig.lineWidth = 2.0f;

    std::vector<std::shared_ptr<WireframePC>> wireframePushConstants(vulkano.scene.nodes.size());
    vul::Vulkano::RenderData wireframeRenderData = defaultRenderData;
    wireframeRenderData.enable = false;
    wireframeRenderData.pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "wireframe.vert.spv", "wireframe.frag.spv", wireframePipelineConfig);
    wireframeRenderData.vertexBuffers = {vulkano.scene.vertexBuffer.get()};
    for (size_t i = 0; i < vulkano.scene.nodes.size(); i++) {
        wireframePushConstants[i] = std::make_shared<WireframePC>();
        wireframePushConstants[i]->wireframeColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        wireframeRenderData.drawDatas[i].pPushData = static_cast<std::shared_ptr<void>>(wireframePushConstants[i]);
        wireframeRenderData.drawDatas[i].pushDataSize = sizeof(WireframePC);
    }
    vulkano.renderDatas.push_back(defaultRenderData);
    vulkano.renderDatas.push_back(wireframeRenderData);

    for (const vul::GltfLoader::GltfNode &node : vulkano.scene.nodes) {
        const vul::GltfLoader::GltfPrimMesh &mesh = vulkano.scene.meshes[node.primMesh];
        for (uint32_t i = mesh.vertexOffset; i < mesh.vertexOffset + mesh.vertexCount; i++) {
            vulkano.scene.vertices[i] = glm::vec3(node.worldMatrix * glm::vec4(vulkano.scene.vertices[i], 1.0f));
        }
    }
    vulkano.scene.vertexBuffer->writeVector(vulkano.scene.vertices, 0);

    std::vector<vul::Vulkano::Descriptor> descs;
    vul::Vulkano::Descriptor desc;
    desc.type = vul::Vulkano::DescriptorType::ssbo;
    desc.stages = {vul::Vulkano::ShaderStage::comp};
    desc.count = 1;
    desc.content = vulkano.scene.indexBuffer.get();
    descs.push_back(desc);
    desc.content = vulkano.scene.vertexBuffer.get();
    descs.push_back(desc);
    desc.content = vulkano.scene.normalBuffer.get();
    descs.push_back(desc);
    renderingResources.normalsUpdaterDescSet = vulkano.createDescriptorSet(descs);

    renderingResources.normalsUpdaterPipeline = std::make_unique<vul::VulCompPipeline>("normalsUpdater.comp.spv", std::vector{renderingResources.normalsUpdaterDescSet->getLayout()->getDescriptorSetLayout()}, vulkano.getVulDevice(), 1);

    return renderingResources;
}

void getRenderingStuffFromObj(RenderingResources &renderingResources, const Obj &obj, vul::Vulkano &vulkano)
{
    std::vector<glm::vec3> simMeshVertices;
    for (const Pointmass &pm : obj.pointMasses) simMeshVertices.push_back(pm.pos);
    std::vector<uint32_t> simMeshIndices;
    for (const VolConstraint &constraint : obj.volConstraints) {
        simMeshIndices.push_back(constraint.pm1idx);
        simMeshIndices.push_back(constraint.pm2idx);
        simMeshIndices.push_back(constraint.pm3idx);

        simMeshIndices.push_back(constraint.pm1idx);
        simMeshIndices.push_back(constraint.pm2idx);
        simMeshIndices.push_back(constraint.pm4idx);

        simMeshIndices.push_back(constraint.pm2idx);
        simMeshIndices.push_back(constraint.pm3idx);
        simMeshIndices.push_back(constraint.pm4idx);

        simMeshIndices.push_back(constraint.pm1idx);
        simMeshIndices.push_back(constraint.pm4idx);
        simMeshIndices.push_back(constraint.pm3idx);
    }
    std::vector<uint32_t> simMeshPointIndices(obj.pointMasses.size());
    for (size_t i = 0; i < simMeshPointIndices.size(); i++) simMeshPointIndices[i] = i;
    std::vector<uint32_t> facetSegmentIndices;
    for (const std::vector<glm::uvec2> &segments : obj.facetSegments) {
        for (glm::uvec2 indices : segments) {
            facetSegmentIndices.push_back(indices.x);
            facetSegmentIndices.push_back(indices.y);
        }
    }

    std::shared_ptr<WireframePC> pushConstant = std::make_shared<WireframePC>();
    pushConstant->wireframeColor = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

    std::shared_ptr<PointsPC> pointPushConstant = std::make_shared<PointsPC>();
    pointPushConstant->pointsColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    pointPushConstant->pointSize = 10.0f;

    std::shared_ptr<WireframePC> facetSegmentsPushConstant = std::make_shared<WireframePC>();
    facetSegmentsPushConstant->wireframeColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    vul::VulPipeline::DrawData drawData{};
    drawData.pPushData = static_cast<std::shared_ptr<void>>(pushConstant);
    drawData.pushDataSize = sizeof(WireframePC);
    drawData.vertexOffset = 0;
    drawData.firstIndex = 0;
    drawData.indexCount = simMeshIndices.size();

    vul::VulPipeline::DrawData pointDrawData = drawData;
    pointDrawData.pPushData = static_cast<std::shared_ptr<void>>(pointPushConstant);
    pointDrawData.pushDataSize = sizeof(PointsPC);
    pointDrawData.indexCount = simMeshPointIndices.size();

    vul::VulPipeline::DrawData facetSegmentsDrawData = drawData;
    facetSegmentsDrawData.pPushData = static_cast<std::shared_ptr<void>>(facetSegmentsPushConstant);
    facetSegmentsDrawData.pushDataSize = sizeof(WireframePC);
    facetSegmentsDrawData.indexCount = facetSegmentIndices.size();

    renderingResources.simMeshVertexBuffer = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
    renderingResources.simMeshVertexBuffer->loadVector(simMeshVertices);
    renderingResources.simMeshVertexBuffer->createBuffer(false, vul::VulBuffer::usage_vertexBuffer);
    renderingResources.simMeshIndexBuffer = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
    renderingResources.simMeshIndexBuffer->loadVector(simMeshIndices);
    renderingResources.simMeshIndexBuffer->createBuffer(true, static_cast<vul::VulBuffer::Usage>(vul::VulBuffer::usage_indexBuffer | vul::VulBuffer::usage_transferDst));
    renderingResources.simMeshPointsIndexBuffer = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
    renderingResources.simMeshPointsIndexBuffer->loadVector(simMeshPointIndices);
    renderingResources.simMeshPointsIndexBuffer->createBuffer(true, static_cast<vul::VulBuffer::Usage>(vul::VulBuffer::usage_indexBuffer | vul::VulBuffer::usage_transferDst));
    renderingResources.facetSegmentsIndexBuffer = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
    renderingResources.facetSegmentsIndexBuffer->loadVector(facetSegmentIndices);
    renderingResources.facetSegmentsIndexBuffer->createBuffer(true, static_cast<vul::VulBuffer::Usage>(vul::VulBuffer::usage_indexBuffer | vul::VulBuffer::usage_transferDst));

    vul::VulPipeline::PipelineConfigInfo pipelineConfig{};
    pipelineConfig.depthAttachmentFormat = vulkano.vulRenderer.getDepthFormat();
    pipelineConfig.colorAttachmentFormats = {vulkano.vulRenderer.getSwapChainColorFormat()};
    pipelineConfig.setLayouts = {vulkano.renderDatas[0].descriptorSets[0][0]->getLayout()->getDescriptorSetLayout()};
    pipelineConfig.bindingDescriptions = {{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};
    pipelineConfig.attributeDescriptions = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    pipelineConfig.polygonMode = VK_POLYGON_MODE_LINE;
    pipelineConfig.lineWidth = 2.0f;

    vul::VulPipeline::PipelineConfigInfo pointsPipelineConfig = pipelineConfig;
    pointsPipelineConfig.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    pointsPipelineConfig.polygonMode = VK_POLYGON_MODE_FILL;

    vul::VulPipeline::PipelineConfigInfo facetSegmentsPipelineConfig = pipelineConfig;
    facetSegmentsPipelineConfig.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    pointsPipelineConfig.polygonMode = VK_POLYGON_MODE_FILL;

    vul::Vulkano::RenderData renderData{};
    renderData.enable = false;
    renderData.pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "wireframe.vert.spv", "wireframe.frag.spv", pipelineConfig);
    renderData.is3d = true;
    renderData.sampleFromDepth = false;
    renderData.depthImageMode = vul::VulRenderer::DepthImageMode::clearPreviousDiscardCurrent;
    renderData.swapChainImageMode = vul::VulRenderer::SwapChainImageMode::clearPreviousStoreCurrent;
    renderData.swapChainClearColor = glm::vec4(0.529f, 0.808f, 0.922f, 1.0f);
    renderData.depthClearColor = 1.0f;
    renderData.indexBuffer = renderingResources.simMeshIndexBuffer.get();
    renderData.vertexBuffers = {renderingResources.simMeshVertexBuffer.get()};
    renderData.descriptorSets = vulkano.renderDatas[0].descriptorSets;
    renderData.drawDatas.push_back(drawData);
    vulkano.renderDatas.push_back(renderData);

    vul::Vulkano::RenderData pointsRenderData = renderData;
    pointsRenderData.enable = false;
    pointsRenderData.pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "points.vert.spv", "points.frag.spv", pointsPipelineConfig);
    pointsRenderData.indexBuffer = renderingResources.simMeshPointsIndexBuffer.get();
    pointsRenderData.drawDatas[0] = pointDrawData;
    vulkano.renderDatas.push_back(pointsRenderData);

    vul::Vulkano::RenderData facetSegmentsRenderData = renderData;
    facetSegmentsRenderData.enable = true;
    facetSegmentsRenderData.pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "wireframe.vert.spv", "wireframe.frag.spv", facetSegmentsPipelineConfig);
    facetSegmentsRenderData.indexBuffer = renderingResources.facetSegmentsIndexBuffer.get();
    facetSegmentsRenderData.drawDatas[0] = facetSegmentsDrawData;
    vulkano.renderDatas.push_back(facetSegmentsRenderData);
}

RenderResult render(vul::Vulkano &vulkano, RenderingResources &renderingResources, const Obj &obj, const Energies &origEnergies)
{
    RenderResult result{false, false, false, 0.0, 0.0};

    VkCommandBuffer cmdBuf = vulkano.startFrame();
    if (cmdBuf == VK_NULL_HANDLE) {
        result.skipFrame = true;
        return result;
    }
    result.frameTime = vulkano.getFrameTime();

    ImGui::Begin("Menu");
    ImGui::Text("Fps: %lf", 1.0 / result.frameTime);
    ImGui::Text("Simulations per second: %lf", static_cast<double>(renderingResources.simsPerFrame) / static_cast<double>(renderingResources.timeSpeed) / result.frameTime);
    ImGui::DragInt("Simulations per frame", &renderingResources.simsPerFrame, static_cast<float>(renderingResources.simsPerFrame) * 0.05, 1, std::numeric_limits<int>::max());
    ImGui::DragFloat("Time speed", &renderingResources.timeSpeed, renderingResources.timeSpeed * 0.05, 0.001, 1000.0);
    ImGui::Text("Potential energy: %lfJ\nKinetic energy: %lfJ\nConstraint energy: %lfJ\nVolume energy: %lfJ\nTotal energy: %lfJ", obj.energies.potentialEnergy, obj.energies.kineticEnergy,
            obj.energies.dstConstraintEnergy, obj.energies.volConstraintEnergy, obj.energies.potentialEnergy + obj.energies.kineticEnergy + obj.energies.dstConstraintEnergy + obj.energies.volConstraintEnergy);
    ImGui::Text("Starting energy: %lfJ", origEnergies.potentialEnergy + origEnergies.kineticEnergy + origEnergies.dstConstraintEnergy + origEnergies.volConstraintEnergy);
    ImGui::Checkbox("Draw wireframes", &vulkano.renderDatas[1].enable);
    ImGui::Checkbox("Fill triangles", &vulkano.renderDatas[0].enable);
    ImGui::Checkbox("Draw simulation mesh", &vulkano.renderDatas[2].enable);
    if (vulkano.renderDatas[2].enable) ImGui::Checkbox("Draw individual simulation tetrahedrons", &renderingResources.drawIndividualTetrahedrons);
    if (renderingResources.drawIndividualTetrahedrons && vulkano.renderDatas[2].enable) {
        assert(vulkano.renderDatas[2].drawDatas[0].indexCount % 12 == 0);
        static int tetrahedronIndex = 0;
        ImGui::SliderInt("Tetrehedron index", &tetrahedronIndex, 0, vulkano.renderDatas[2].drawDatas[0].indexCount / 12 - 1);
        vulkano.renderDatas[2].drawDatas[0].firstIndex = tetrahedronIndex * 12;
    } else vulkano.renderDatas[2].drawDatas[0].firstIndex = 0;
    ImGui::Checkbox("Draw simulation mesh points", &vulkano.renderDatas[3].enable);
    if (vulkano.renderDatas[3].enable) {
        PointsPC *pointsPC = static_cast<PointsPC *>(vulkano.renderDatas[3].drawDatas[0].pPushData.get());
        ImGui::DragFloat("Point size", &pointsPC->pointSize, 0.05, 0.0, 1000.0);
    }
    ImGui::Checkbox("Draw facet segments", &vulkano.renderDatas[4].enable);
    if (vulkano.renderDatas[4].enable) {
        static bool drawIndividualFacets = false;
        static const uint32_t maxFacetIndexCount = vulkano.renderDatas[4].drawDatas[0].indexCount;
        ImGui::Checkbox("Draw individual facets", &drawIndividualFacets);
        if (drawIndividualFacets) {
            static int facetIndex = 0;
            ImGui::SliderInt("Facet index", &facetIndex, 0, obj.facetSegments.size() - 1);
            uint32_t fistIndex = 0;
            for (int i = 0; i < facetIndex; i++) fistIndex += obj.facetSegments[i].size();
            vulkano.renderDatas[4].drawDatas[0].firstIndex = fistIndex * 2;
            vulkano.renderDatas[4].drawDatas[0].indexCount = obj.facetSegments[facetIndex].size() * 2;
            ImGui::Text("Facet segment count: %u", vulkano.renderDatas[4].drawDatas[0].indexCount / 2);
        } else {
            vulkano.renderDatas[4].drawDatas[0].firstIndex = 0;
            vulkano.renderDatas[4].drawDatas[0].indexCount = maxFacetIndexCount;
        }
    }
    ImGui::Checkbox("Simulate", &renderingResources.simulate);
    result.reset = ImGui::Button("Reset");
    ImGui::End();

    Ubo ubo;
    ubo.projViewMat = vulkano.camera.getProjection() * vulkano.camera.getView();
    ubo.camPos = glm::vec4(vulkano.cameraTransform.pos, 1.0f);
    ubo.lightPos = glm::vec4(vulkano.scene.lights[0].position, vulkano.scene.lights[0].range);
    ubo.lightColor = glm::vec4(vulkano.scene.lights[0].color, vulkano.scene.lights[0].intensity);
    ubo.ambientLightColor = glm::vec4(vulkano.renderDatas[0].swapChainClearColor);
    renderingResources.ubos[vulkano.getFrameIdx()]->writeData(&ubo, sizeof(ubo), 0);

    if (renderingResources.simulate) {
        for (uint32_t i = 0; i < obj.mesh.vertexCount; i++)
            vulkano.scene.vertices[obj.mesh.vertexOffset + i] = obj.pointMasses[obj.meshVertexIdxToPointMassIdx.at(i)].pos;
        vulkano.scene.vertexBuffer->writeData(&vulkano.scene.vertices[obj.mesh.vertexOffset], obj.mesh.vertexCount * sizeof(glm::vec3), obj.mesh.vertexOffset * sizeof(glm::vec3));
        std::vector<glm::vec3> simMeshVertices;
        for (const Pointmass &pm : obj.pointMasses) simMeshVertices.push_back(pm.pos);
        renderingResources.simMeshVertexBuffer->writeVector(simMeshVertices, 0);
    }

    const uint32_t indexCount = vulkano.renderDatas[2].drawDatas[0].indexCount;
    if (renderingResources.drawIndividualTetrahedrons) vulkano.renderDatas[2].drawDatas[0].indexCount = 12;
    result.exit = vulkano.endFrame(cmdBuf);
    if (renderingResources.drawIndividualTetrahedrons) vulkano.renderDatas[2].drawDatas[0].indexCount = indexCount;

    /*
    The normalsUpdater isn't quite ready yet.

    NormalUpdaterPC normalUpdaterPC;
    normalUpdaterPC.triangleCount = obj.mesh.indexCount / 3;
    normalUpdaterPC.indexOffset = obj.mesh.firstIndex / 3;
    normalUpdaterPC.vertexOffset = obj.mesh.vertexOffset;
    renderingResources.normalsUpdaterPipeline->pPushData = &normalUpdaterPC;
    renderingResources.normalsUpdaterPipeline->pushSize = sizeof(normalUpdaterPC);
    renderingResources.normalsUpdaterPipeline->begin({renderingResources.normalsUpdaterDescSet->getSet()});
    renderingResources.normalsUpdaterPipeline->dispatch(normalUpdaterPC.triangleCount / 32, 1, 1);
    renderingResources.normalsUpdaterPipeline->end(false);
    */

    result.frameTime *= renderingResources.timeSpeed;
    result.simDeltaT = result.frameTime / static_cast<double>(renderingResources.simsPerFrame);
    
    return result;
}
