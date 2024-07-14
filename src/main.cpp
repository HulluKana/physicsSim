#include "host_device.hpp"
#include "imgui.h"
#include "vul_buffer.hpp"
#include "vul_descriptors.hpp"
#include "vul_gltf_loader.hpp"
#include "vul_pipeline.hpp"
#include "vul_renderer.hpp"
#include "vul_settings.hpp"
#include "vul_swap_chain.hpp"
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <vulkano_program.hpp>

int main()
{
    vul::Vulkano vulkano(2560, 1440, "Physics simulator");
    vul::settings::maxFps = 144.0;

    vulkano.loadScene("../Models/Physic sim scene.gltf", "../Models", vul::Scene::WantedBuffers{.vertex = true, .index = true, .normal = true, .tangent = false, .uv = false, .material = true, .primInfo = false});
    std::vector<std::shared_ptr<vul::VulDescriptorSet>> descSets(vul::VulSwapChain::MAX_FRAMES_IN_FLIGHT);
    std::vector<std::unique_ptr<vul::VulBuffer>> ubos(descSets.size());
    for (size_t i = 0; i < descSets.size(); i++) {
        ubos[i] = std::make_unique<vul::VulBuffer>(vulkano.getVulDevice());
        ubos[i]->keepEmpty(sizeof(Ubo), 1);
        ubos[i]->createBuffer(false, vul::VulBuffer::usage_ubo);

        std::vector<vul::Vulkano::Descriptor> descs;
        vul::Vulkano::Descriptor desc;
        desc.type = vul::Vulkano::DescriptorType::ubo;
        desc.stages = {vul::Vulkano::ShaderStage::vert, vul::Vulkano::ShaderStage::frag};
        desc.count = 1;
        desc.content = ubos[i].get();
        descs.push_back(desc);

        desc.type = vul::Vulkano::DescriptorType::ssbo;
        desc.stages = {vul::Vulkano::ShaderStage::frag};
        desc.content = vulkano.scene.materialBuffer.get();
        descs.push_back(desc);

        descSets[i] = vulkano.createDescriptorSet(descs);
    }

    vul::VulPipeline::PipelineConfigInfo pipelineConfig{};
    pipelineConfig.depthAttachmentFormat = vulkano.vulRenderer.getDepthFormat();
    pipelineConfig.colorAttachmentFormats = {vulkano.vulRenderer.getSwapChainColorFormat()};
    pipelineConfig.setLayouts = {descSets[0]->getLayout()->getDescriptorSetLayout()};
    pipelineConfig.bindingDescriptions = {{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}, {1, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};
    pipelineConfig.attributeDescriptions = {{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0}};
    std::shared_ptr<vul::VulPipeline> pipeline = std::make_shared<vul::VulPipeline>(vulkano.getVulDevice(), "3D.vert.spv", "3D.frag.spv", pipelineConfig);

    std::vector<std::shared_ptr<PushConstant>> pushConstants(vulkano.scene.nodes.size());
    vul::Vulkano::RenderData renderData{};
    renderData.enable = true;
    renderData.pipeline = pipeline;
    renderData.is3d = true;
    renderData.sampleFromDepth = false;
    renderData.depthImageMode = vul::VulRenderer::DepthImageMode::clearPreviousDiscardCurrent;
    renderData.swapChainImageMode = vul::VulRenderer::SwapChainImageMode::clearPreviousStoreCurrent;
    renderData.swapChainClearColor = glm::vec4(0.529f, 0.808f, 0.922f, 1.0f);
    renderData.depthClearColor = 1.0f;
    for (int i = 0; i < vul::VulSwapChain::MAX_FRAMES_IN_FLIGHT; i++) renderData.descriptorSets[i] = {descSets[i]};
    for (size_t i = 0; i < vulkano.scene.nodes.size(); i++) {
        const vul::GltfLoader::GltfNode &node = vulkano.scene.nodes[i];
        const vul::GltfLoader::GltfPrimMesh &mesh = vulkano.scene.meshes[node.primMesh];
        pushConstants[i] = std::make_shared<PushConstant>();
        pushConstants[i]->modelMat = node.worldMatrix;
        pushConstants[i]->matIdx = mesh.materialIndex;


        vul::VulPipeline::DrawData drawData{};
        drawData.pPushData = static_cast<std::shared_ptr<void>>(pushConstants[i]);
        drawData.pushDataSize = sizeof(PushConstant);
        drawData.vertexOffset = mesh.vertexOffset;
        drawData.firstIndex = mesh.firstIndex;
        drawData.indexCount = mesh.indexCount;
        renderData.drawDatas.push_back(drawData);
    }
    vulkano.renderDatas.push_back(renderData);
    
    bool stop = false;
    while (!stop) {
        VkCommandBuffer cmdBuf = vulkano.startFrame();
        if (cmdBuf == VK_NULL_HANDLE) continue;

        Ubo ubo;
        ubo.projViewMat = vulkano.camera.getProjection() * vulkano.camera.getView();
        ubo.camPos = glm::vec4(vulkano.cameraTransform.pos, 1.0f);
        ubo.lightPos = glm::vec4(vulkano.scene.lights[0].position, vulkano.scene.lights[0].range);
        ubo.lightColor = glm::vec4(vulkano.scene.lights[0].color, vulkano.scene.lights[0].intensity);
        ubo.ambientLightColor = glm::vec4(renderData.swapChainClearColor);
        ubos[vulkano.getFrameIdx()]->writeData(&ubo, sizeof(ubo), 0);

        ImGui::Begin("Performance");
        ImGui::Text("Fps: %f", 1.0f / vulkano.getFrameTime());
        ImGui::End();

        stop = vulkano.endFrame(cmdBuf);
    }

    vulkano.letVulkanoFinish();
}
