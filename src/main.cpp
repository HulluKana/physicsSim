#include "vul_settings.hpp"
#include <rendering.hpp>
#include <simulation.hpp>
#include <imgui.h>
#include <memory>
#include <vulkano_program.hpp>
#include <iostream>

int main()
{
    vul::Vulkano vulkano(2560, 1440, "Physics simulator");
    vul::settings::maxFps = 144.0;
    vul::settings::cameraProperties.nearPlane = 0.005f;
    vulkano.cameraController.baseMoveSpeed = 1.5f;
    vulkano.cameraController.speedChanger = 5.0f;
    RenderingResources renderingResources = initializeRenderingStuff("../Models", "Physic sim scene.gltf", vulkano);
    renderingResources.simsPerFrame = 500;
    renderingResources.timeSpeed = 1.0;
    renderingResources.simulate = false;
    renderingResources.drawIndividualTetrahedrons = false;
    Obj origObj = getObjFromScene(vulkano.scene, "Wheel");
    Obj obj = origObj;
    getRenderingStuffFromObj(renderingResources, obj, vulkano);
    while (true) {
        RenderResult renderingResult = render(vulkano, renderingResources, obj, origObj.energies);
        if (renderingResult.exit) break;
        if (renderingResult.skipFrame) continue;
        if (renderingResult.reset) obj = origObj;
        if (renderingResources.simulate) simulate(obj, renderingResult.frameTime, renderingResult.simDeltaT);
    }
    vulkano.letVulkanoFinish();
}
