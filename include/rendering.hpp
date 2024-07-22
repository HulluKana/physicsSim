#pragma once

#include "simulation.hpp"
#include "vulkano_program.hpp"
#include <array>
struct RenderingResources {
    std::array<std::unique_ptr<vul::VulBuffer>, vul::VulSwapChain::MAX_FRAMES_IN_FLIGHT> ubos;
    std::unique_ptr<vul::VulBuffer> simMeshIndexBuffer;
    std::unique_ptr<vul::VulBuffer> simMeshVertexBuffer;
    int simsPerFrame;
    bool simulate;
    bool drawIndividualTetrahedrons;
};
struct RenderResult {
    bool skipFrame;
    bool exit;
    bool reset;
    double frameTime;
    double simDeltaT;
};

RenderingResources initializeRenderingStuff(const std::string &modelDir, const std::string &modelFileName, vul::Vulkano &vulkano);
void getRenderingStuffFromObj(RenderingResources &renderingResources, const Obj &obj, vul::Vulkano &vulkano);
RenderResult render(vul::Vulkano &vulkano, RenderingResources &renderingResources, const Obj &obj, const Energies &origEnergies);
