#include "simulation.hpp"
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
    defaultRenderData.enable = true;
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
    wireframeRenderData.pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "wireframe.vert.spv", "wireframe.frag.spv", wireframePipelineConfig);
    wireframeRenderData.vertexBuffers = {vulkano.scene.vertexBuffer.get()};
    for (size_t i = 0; i < vulkano.scene.nodes.size(); i++) {
        wireframePushConstants[i] = std::make_shared<WireframePC>();
        wireframePushConstants[i]->wireframeColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        wireframeRenderData.drawDatas[i].pPushData = static_cast<std::shared_ptr<void>>(wireframePushConstants[i]);
        wireframeRenderData.drawDatas[i].pushDataSize = sizeof(DefaultPC);
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

    return renderingResources;
}

void getRenderingStuffFromObj(RenderingResources &renderingResources, const Obj &obj, vul::Vulkano &vulkano)
{
    std::vector<glm::vec3> simMeshVertices;
    for (const Pointmass &pm : obj.pointMasses) simMeshVertices.push_back(pm.pos);
    std::vector<uint32_t> simMeshIndices;
    for (const DstConstraint &constraint : obj.dstConstraints) {
        simMeshIndices.push_back(constraint.pm1idx);
        simMeshIndices.push_back(constraint.pm2idx);
    }

    std::shared_ptr<WireframePC> pushConstant = std::make_shared<WireframePC>();
    pushConstant->wireframeColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);

    vul::VulPipeline::DrawData drawData{};
    drawData.pPushData = static_cast<std::shared_ptr<void>>(pushConstant);
    drawData.pushDataSize = sizeof(DefaultPC);
    drawData.vertexOffset = 0;
    drawData.firstIndex = 0;
    drawData.indexCount = simMeshIndices.size();

    renderingResources.simMeshVertexBuffer = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
    renderingResources.simMeshVertexBuffer->loadVector(simMeshVertices);
    renderingResources.simMeshVertexBuffer->createBuffer(false, vul::VulBuffer::usage_vertexBuffer);
    renderingResources.simMeshIndexBuffer = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
    renderingResources.simMeshIndexBuffer->loadVector(simMeshIndices);
    renderingResources.simMeshIndexBuffer->createBuffer(true, static_cast<vul::VulBuffer::Usage>(vul::VulBuffer::usage_indexBuffer | vul::VulBuffer::usage_transferDst));

    vul::VulPipeline::PipelineConfigInfo pipelineConfig{};
    pipelineConfig.depthAttachmentFormat = vulkano.vulRenderer.getDepthFormat();
    pipelineConfig.colorAttachmentFormats = {vulkano.vulRenderer.getSwapChainColorFormat()};
    pipelineConfig.setLayouts = {vulkano.renderDatas[0].descriptorSets[0][0]->getLayout()->getDescriptorSetLayout()};
    pipelineConfig.bindingDescriptions = {{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};
    pipelineConfig.attributeDescriptions = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    pipelineConfig.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    pipelineConfig.lineWidth = 2.0f;


    vul::Vulkano::RenderData renderData{};
    renderData.enable = true;
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
    ImGui::DragInt("Simulations per frame", &renderingResources.simsPerFrame, static_cast<float>(renderingResources.simsPerFrame) * 0.05, 1, std::numeric_limits<int>::max());
    ImGui::Text("Potential energy: %lfJ\nKinetic energy: %lfJ\nConstraint energy: %lfJ\nVolume energy: %lfJ\nTotal energy: %lfJ", obj.energies.potentialEnergy, obj.energies.kineticEnergy,
            obj.energies.dstConstraintEnergy, obj.energies.volConstraintEnergy, obj.energies.potentialEnergy + obj.energies.kineticEnergy + obj.energies.dstConstraintEnergy + obj.energies.volConstraintEnergy);
    ImGui::Text("Starting energy: %lfJ", origEnergies.potentialEnergy + origEnergies.kineticEnergy + origEnergies.dstConstraintEnergy + origEnergies.volConstraintEnergy);
    ImGui::Checkbox("Draw wireframes", &vulkano.renderDatas[1].enable);
    ImGui::Checkbox("Fill triangles", &vulkano.renderDatas[0].enable);
    ImGui::Checkbox("Draw simulation mesh", &vulkano.renderDatas[2].enable);
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

    result.exit = vulkano.endFrame(cmdBuf);
    result.simDeltaT = result.frameTime / static_cast<double>(renderingResources.simsPerFrame);
    
    return result;
}
